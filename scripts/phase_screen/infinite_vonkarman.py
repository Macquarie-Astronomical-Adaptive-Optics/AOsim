# CuPy-only "infinite phase screen" + layered AO sampling
#
# Key idea: keep NxN window and extend by generating new rows/cols:
#   X = A Z + B b
# where A, B derived from covariance matrices.
# based on  
# François Assémat, Richard W. Wilson, and Eric Gendron, 
# "Method for simulating infinitely long and non stationary phase screens with optimized memory storage," 
# Opt. Express 14, 988-999 (2006) 
#
# To allow for arbitrary wind direction (vx, vy)
# create prepend and postpend versions of A and B

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cupy as cp
from cupyx.scipy.special import gamma  # GPU special functions

ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)

from . import kv

# ---------- von Karman phase covariance ----------
def phase_covariance_vk_cp(r: cp.ndarray, r0: float, L0: float, dtype=cp.float32) -> cp.ndarray:
    r = cp.asarray(r, dtype=dtype)
    r0 = float(r0)
    L0 = float(L0)

    # avoid r=0 singular behaviour
    r = r + dtype(1e-40)

    A = (L0 / r0) ** (5.0 / 3.0)
    B1 = (2.0 ** (-5.0 / 6.0)) * gamma(11.0 / 6.0) / (cp.pi ** (8.0 / 3.0))
    B2 = ((24.0 / 5.0) * gamma(6.0 / 5.0)) ** (5.0 / 6.0)

    x = (2.0 * cp.pi * r) / L0
    C = (x ** (5.0 / 6.0)) * kv.kv_realpos(5.0 / 6.0, x)

    return (A * B1 * B2 * C).astype(dtype)


# ---------- FFT initialisation (CuPy) ----------
def _ft_phase_screen_vk_cp(N: int, dx: float, r0: float, L0: float, seed: int, dtype=cp.float32) -> cp.ndarray:
    """
    Initial screen (periodic) via FFT-based von Karman PSD.
    This is only to seed the window; continued extension removes "wrap repeat" behaviour over time.
    """
    N = int(N)
    dx = float(dx)
    r0 = float(r0)
    L0 = float(L0)

    cdtype = cp.complex64 if dtype == cp.float32 else cp.complex128
    rng = cp.random.default_rng(int(seed))

    fx = cp.fft.fftfreq(N, d=dx).astype(dtype)  # cycles/m
    fy = cp.fft.fftfreq(N, d=dx).astype(dtype)
    FX, FY = cp.meshgrid(fx, fy, indexing="xy")
    f2 = FX * FX + FY * FY

    # von Karman phase PSD ~ 0.023 r0^(-5/3) (f^2 + 1/L0^2)^(-11/6)
    eps = dtype(1e-30)
    PSD = (dtype(0.023) * (r0 ** (-5.0 / 3.0)) * ((f2 + (1.0 / L0) ** 2 + eps) ** (-11.0 / 6.0))).astype(dtype)
    PSD[0, 0] = dtype(0.0)

    real = rng.standard_normal((N, N), dtype=dtype)
    imag = rng.standard_normal((N, N), dtype=dtype)
    white = (real + 1j * imag).astype(cdtype)

    # enforce Hermitian symmetry so IFFT is real
    white = 0.5 * (white + cp.conj(cp.roll(cp.roll(white[::-1, ::-1], 1, axis=0), 1, axis=1)))

    Ak = (white * cp.sqrt(PSD).astype(cdtype)).astype(cdtype)
    phi = cp.fft.ifft2(Ak).real.astype(dtype)
    phi -= cp.mean(phi)
    return phi


# ---------- Core infinite screen window ----------
class InfiniteVonKarmanScreen2D:
    """
    Non-periodic, infinite-in-extent (by extension) von Karman screen.
    Keeps only NxN window, extends by generating new rows/cols:
      X = A Z + B b 

    Supports shifting in BOTH x and y by extending edges as needed.
    """

    _AB_CACHE = {}  # key -> (A_ref, B_ref) built for r0_ref = 1.0

    def __init__(self, N, dx, r0, L0, seed=0, n_columns=2, dtype=cp.float32):
        self.N = int(N)
        self.dx = float(dx)
        self.r0 = float(r0)
        self.L0 = float(L0)
        self.seed = int(seed)
        self.n_columns = int(n_columns)
        self.dtype = dtype
        self._rng = cp.random.default_rng(self.seed)

        self._frac_x = 0.0
        self._frac_y = 0.0

        self.origin_x = 0.0
        self.origin_y = 0.0

        # Build/reuse A_ref,B_ref for this geometry+L0
        self._load_or_build_AB_cached()

        # Scale B for this layer's r0:
        # cov ∝ r0^{-5/3} => amplitude ∝ r0^{-5/6}
        amp = (self.r0 ** (-5.0/6.0))
        self.A_pre  = self.A_pre_ref
        self.A_post = self.A_post_ref
        self.B_pre  = (self.B_pre_ref  * self.dtype(amp)).astype(self.dtype)
        self.B_post = (self.B_post_ref * self.dtype(amp)).astype(self.dtype)

        # Initial window (seeded) – generate for r0_ref=1 then scale by r0^{-5/6}
        self.scrn = _ft_phase_screen_vk_cp(self.N, self.dx, r0=1.0, L0=self.L0, seed=self.seed + 1337, dtype=self.dtype)
        self.scrn *= self.dtype(amp)
        self.scrn -= cp.mean(self.scrn)
        
        self.warmup()

    def _cache_key(self):
        return (self.N, float(self.dx), float(self.L0), int(self.n_columns), str(self.dtype))

    def _cache_key(self):
        return (self.N, float(self.dx), float(self.L0), int(self.n_columns), str(self.dtype))

    def _load_or_build_AB_cached(self):
        key = self._cache_key()
        cached = self._AB_CACHE.get(key, None)
        if cached is not None:
            self.A_pre_ref, self.B_pre_ref, self.A_post_ref, self.B_post_ref = cached
            return

        Apre, Bpre, Apost, Bpost = self._build_AB_pair_for_r0(r0_ref=1.0, L0=self.L0)
        self._AB_CACHE[key] = (Apre, Bpre, Apost, Bpost)
        self.A_pre_ref, self.B_pre_ref, self.A_post_ref, self.B_post_ref = Apre, Bpre, Apost, Bpost


    def _build_AB_pair_for_r0(self, r0_ref: float, L0: float):
        """
        Build operator pairs:
        - (A_pre, B_pre): new row at x = -1 (prepend)
        - (A_post,B_post): new row at x = n_columns (append)
        """
        N  = self.N
        nc = self.n_columns
        dx = self.dx

        build_dtype = cp.float64  # stability

        # stencil points: x=0..nc-1, y=0..N-1
        xs = cp.repeat(cp.arange(nc, dtype=build_dtype), N)
        ys = cp.tile(cp.arange(N, dtype=build_dtype), nc)
        pos_stencil = cp.stack([xs, ys], axis=1) * dx         # (nc*N,2)
        n_stencils = nc * N

        def build_ops(x_new: float):
            pos_X = cp.stack(
                [cp.full((N,), x_new, dtype=build_dtype), cp.arange(N, dtype=build_dtype)],
                axis=1
            ) * dx                                             # (N,2)

            pos = cp.concatenate([pos_stencil, pos_X], axis=0) # (P,2)
            d = pos[:, None, :] - pos[None, :, :]
            r = cp.sqrt(d[..., 0]**2 + d[..., 1]**2).astype(build_dtype)

            cov = phase_covariance_vk_cp(r, r0=float(r0_ref), L0=float(L0), dtype=build_dtype)

            cov_zz = cov[:n_stencils, :n_stencils]
            cov_xx = cov[n_stencils:, n_stencils:]
            cov_zx = cov[:n_stencils, n_stencils:]
            cov_xz = cov[n_stencils:, :n_stencils]

            A = cp.linalg.solve(cov_zz.T, cov_xz.T).T
            BBt = cov_xx - A @ cov_zx

            jitter = (build_dtype(1e-6) * cp.trace(BBt) / build_dtype(N))
            Lchol = cp.linalg.cholesky(BBt + jitter * cp.eye(N, dtype=build_dtype))
            B = Lchol

            return A.astype(self.dtype), B.astype(self.dtype)

        # prepend: new row *before* stencil
        A_pre, B_pre = build_ops(x_new=-1.0)
        # append: new row *after* stencil
        A_post, B_post = build_ops(x_new=float(nc))

        return A_pre, B_pre, A_post, B_post


    # If r0 changes on the fly:
    def rebuild_L0(self, L0_new: float, seed: int | None = None):
        self.L0 = float(L0_new)
        if seed is not None:
            self.seed = int(seed)
            self._rng = cp.random.default_rng(self.seed)

        # rebuild/load cached ref operators for this new L0
        self._load_or_build_AB_cached()

        # rescale B for current r0
        amp = (self.r0 ** (-5.0 / 6.0))
        self.A_pre  = self.A_pre_ref
        self.A_post = self.A_post_ref
        self.B_pre  = (self.B_pre_ref  * self.dtype(amp)).astype(self.dtype)
        self.B_post = (self.B_post_ref * self.dtype(amp)).astype(self.dtype)

        # You can keep the current scrn (recommended: avoids a visible “jump”)
        # but its statistics are now for the old L0. If you want to fully apply new L0,
        # either warmup after changing L0 or reseed:
        self.scrn = _ft_phase_screen_vk_cp(self.N, self.dx, r0=1.0, L0=self.L0, seed=self.seed + 1337, dtype=self.dtype) * self.dtype(amp)
        self.scrn -= cp.mean(self.scrn)


    def rescale_r0(self, r0_new: float):
        r0_new = float(r0_new)
        if r0_new <= 0:
            raise ValueError("r0 must be > 0")
        if r0_new == self.r0:
            return

        # amplitude factor amp = r0^{-5/6}
        amp_old = (self.r0 ** (-5.0 / 6.0))
        amp_new = (r0_new ** (-5.0 / 6.0))
        s = amp_new / amp_old

        self.scrn *= self.dtype(s)
        self.B_pre *= self.dtype(s)
        self.B_post *= self.dtype(s)
        self.r0 = r0_new


    # ----- row/col generation -----
    def _new_row(self, edge: str) -> cp.ndarray:
        """
        edge:
        'top'    => prepend (x=-1) using first nc rows
        'bottom' => append  (x=nc) using last  nc rows
        """
        nc = self.n_columns

        if edge == "top":
            stencil = self.scrn[:nc, :].reshape(-1)
            A, B = self.A_pre, self.B_pre
        elif edge == "bottom":
            stencil = self.scrn[-nc:, :].reshape(-1)
            A, B = self.A_post, self.B_post
        else:
            raise ValueError("edge must be 'top' or 'bottom'")

        b = self._rng.standard_normal((self.N,), dtype=self.dtype)
        return (A @ stencil + B @ b).astype(self.dtype)

    def _new_col(self, edge: str) -> cp.ndarray:
        """
        edge:
        'left'  => prepend (x=-1) using first nc cols  (transpose view)
        'right' => append  (x=nc) using last  nc cols
        """
        nc = self.n_columns
        scrT = self.scrn.T  # view

        if edge == "left":
            stencil = scrT[:nc, :].reshape(-1)
            A, B = self.A_pre, self.B_pre
        elif edge == "right":
            stencil = scrT[-nc:, :].reshape(-1)
            A, B = self.A_post, self.B_post
        else:
            raise ValueError("edge must be 'left' or 'right'")

        b = self._rng.standard_normal((self.N,), dtype=self.dtype)
        return (A @ stencil + B @ b).astype(self.dtype)

    def _shift_down_1px(self):
        new = self._new_row("top")                      # new enters at top
        self.scrn[1:, :] = self.scrn[:-1, :].copy()
        self.scrn[0, :] = new

    def _shift_up_1px(self):
        new = self._new_row("bottom")                   # new enters at bottom
        self.scrn[:-1, :] = self.scrn[1:, :].copy()
        self.scrn[-1, :] = new

    def _shift_right_1px(self):
        new = self._new_col("left")                     # new enters at left
        self.scrn[:, 1:] = self.scrn[:, :-1].copy()
        self.scrn[:, 0] = new

    def _shift_left_1px(self):
        new = self._new_col("right")                    # new enters at right
        self.scrn[:, :-1] = self.scrn[:, 1:].copy()
        self.scrn[:, -1] = new

    def _pan_include_left(self):
        self._shift_right_1px()
        self.origin_x -= 1.0

    def _pan_include_right(self):
        self._shift_left_1px()
        self.origin_x += 1.0

    def _pan_include_top(self):
        self._shift_down_1px()
        self.origin_y -= 1.0

    def _pan_include_bottom(self):
        self._shift_up_1px()
        self.origin_y += 1.0

    # ----- time advance -----
    def warmup(self, n_steps: int = None):
        # replace the initial FFT seed with AR-generated content
        if n_steps is None:
            n_steps = 2 * self.N
        for _ in range(int(n_steps)):
            self._shift_right_1px()
            self._shift_down_1px()
        self.scrn -= cp.mean(self.scrn)


    def advance(self, vx: float, vy: float, dt: float):
        """
        Update screen for frozen-flow wind (vx, vy) [m/s] over dt [s].
        Matches the common convention: phi(x,y,t) = phi0(x - vx t, y - vy t),
        i.e. phase pattern moves +vx,+vy across the pupil.
        """
        dx = self.dx
        sx = float(vx) * float(dt) / dx  # pixels
        sy = float(vy) * float(dt) / dx  # pixels

        self._frac_x += sx
        self._frac_y += sy

        # apply integer pixel shifts by extending edges
        while self._frac_x >= 1.0:
            self._shift_right_1px()
            self._frac_x -= 1.0
        while self._frac_x <= -1.0:
            self._shift_left_1px()
            self._frac_x += 1.0

        while self._frac_y >= 1.0:
            self._shift_down_1px()
            self._frac_y -= 1.0
        while self._frac_y <= -1.0:
            self._shift_up_1px()
            self._frac_y += 1.0

    def hard_reset(self):
        # reset RNG + offsets
        self._rng = cp.random.default_rng(self.seed)
        self._frac_x = 0.0
        self._frac_y = 0.0

        # re-seed initial window (using r0_ref=1 then scale)
        amp = (self.r0 ** (-5.0/6.0))
        self.scrn = _ft_phase_screen_vk_cp(self.N, self.dx, r0=1.0, L0=self.L0,
                                        seed=self.seed + 1337, dtype=self.dtype)
        self.scrn *= self.dtype(amp)
        self.scrn -= cp.mean(self.scrn)

        # optional: if you warmup at init, do the same here
        self.warmup()


    # ----- sampling -----
    def ensure_region_world(self, Xw: cp.ndarray, Yw: cp.ndarray, margin: float = 2.0):
        """
        Ensure all requested world coords are inside the local bilinear-safe range [0, N-2]
        after applying (origin + fractional wind shift).
        """
        Xw = cp.asarray(Xw, dtype=self.dtype)
        Yw = cp.asarray(Yw, dtype=self.dtype)

        # Convert world coords to local sample coords (same transform sample uses)
        # sample_bilinear subtracts frac; we include it here so bounds are correct.
        Xloc_min = float(cp.min(Xw) - (self.origin_x + self._frac_x))
        Xloc_max = float(cp.max(Xw) - (self.origin_x + self._frac_x))
        Yloc_min = float(cp.min(Yw) - (self.origin_y + self._frac_y))
        Yloc_max = float(cp.max(Yw) - (self.origin_y + self._frac_y))

        lo = 0.0 + float(margin)
        hi = (self.N - 2.0) - float(margin)

        # Pan until region fits
        while Xloc_min < lo:
            self._pan_include_left()
            Xloc_min += 1.0
            Xloc_max += 1.0

        while Xloc_max > hi:
            self._pan_include_right()
            Xloc_min -= 1.0
            Xloc_max -= 1.0

        while Yloc_min < lo:
            self._pan_include_top()
            Yloc_min += 1.0
            Yloc_max += 1.0

        while Yloc_max > hi:
            self._pan_include_bottom()
            Yloc_min -= 1.0
            Yloc_max -= 1.0

    def sample_bilinear_world(self, Xw: cp.ndarray, Yw: cp.ndarray, autopan: bool = True, margin: float = 2.0):
        """
        Sample using world pixel coords. Optionally pan the cached window so coords are in-bounds.
        """
        Xw = cp.asarray(Xw, dtype=self.dtype)
        Yw = cp.asarray(Yw, dtype=self.dtype)

        if autopan:
            self.ensure_region_world(Xw, Yw, margin=margin)

        # World -> local coords used for bilinear indexing
        X = Xw - self.dtype(self.origin_x + self._frac_x)
        Y = Yw - self.dtype(self.origin_y + self._frac_y)

        x0 = cp.floor(X).astype(cp.int32)
        y0 = cp.floor(Y).astype(cp.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = (X - x0).astype(self.dtype)
        wy = (Y - y0).astype(self.dtype)

        phi = self.scrn
        v00 = phi[y0, x0]
        v10 = phi[y0, x1]
        v01 = phi[y1, x0]
        v11 = phi[y1, x1]

        return (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 + (1 - wx) * wy * v01 + wx * wy * v11


    def view_patch(self, size_pixels, M: int, angle_deg: float = 0.0, margin: float = 2.0, center_xy_pix=None):
        """
        Return a fixed world-anchored patch (for display).
        If center_xy_pix is None, uses the screen center based on self.N.
        """
        if center_xy_pix is None:
            cx = (self.N - 1) / 2.0
            cy = (self.N - 1) / 2.0
        else:
            cx, cy = float(center_xy_pix[0]), float(center_xy_pix[1])

        side = float(size_pixels)
        M = int(M)

        step = side / float(M)
        lin = (cp.arange(M, dtype=self.dtype) - (M - 1) / 2.0) * step
        Xs, Ys = cp.meshgrid(lin, lin, indexing="xy")

        a = math.radians(float(angle_deg))
        ca, sa = math.cos(a), math.sin(a)

        Xw = ca * Xs - sa * Ys + cx
        Yw = sa * Xs + ca * Ys + cy

        return self.sample_bilinear_world(Xw, Yw, autopan=True, margin=margin)



# ---------- Multi-layer wrapper API ----------
@dataclass
class _Layer:
    N: float
    name: str
    altitude_m: float
    r0_layer: float
    L0: float
    wind: Tuple[float, float]
    seed_offset: int
    screen: InfiniteVonKarmanScreen2D
    active: bool = True
    phi: Optional[cp.ndarray] = None # current NxN window


class LayeredInfinitePhaseScreen:

    def __init__(self, N: int, dx: float, seed: int = 0, n_columns: int = 2, dtype=cp.float32, **_):
        self.N = int(N)
        self.dx = float(dx)
        self.seed = int(seed)
        self.n_columns = int(n_columns)
        self.dtype = dtype

        self.layers: List[dict] = []        # dict view (GUI/worker)
        self._layers_obj: List[_Layer] = [] # object view (sampling)

        self.r0_total = None

    def _compute_r0_total(self) -> float:
        s = 0.0
        for L in self._layers_obj:
            if not L.active:
                continue
            s += float(L.r0_layer) ** (-5.0/3.0)
        return float("inf") if s == 0 else s ** (-3.0/5.0)

    def _compute_weights(self):
        # for display/debug only
        vals = [float(L.r0_layer) ** (-5.0 / 3.0) for L in self._layers_obj]
        denom = sum(vals)
        if denom <= 0:
            return [0.0] * len(vals)
        return [v / denom for v in vals]

    def _refresh_totals(self):
        self.r0_total = self._compute_r0_total()
        ws = self._compute_weights()
        for i, w in enumerate(ws):
            self.layers[i]["w"] = float(w)

        return self.r0_total

    def _rebuild_name_map(self):
        self._name_to_idx = {}
        for i, L in enumerate(self._layers_obj):
            self._name_to_idx[L.name] = i

    def _idx_from_name(self, name: str) -> int:
        if not self._name_to_idx:
            self._rebuild_name_map()
        if name not in self._name_to_idx:
            # fallback: rebuild in case names changed
            self._rebuild_name_map()
        if name not in self._name_to_idx:
            raise KeyError(f"Unknown layer name: {name!r}. Available: {list(self._name_to_idx.keys())}")
        return int(self._name_to_idx[name])

    def add_layer(self, name: str, altitude_m: float, r0: float,
              L0: float = 30.0, wind: Sequence[float] = (0.0, 0.0),
              seed_offset: int = 0, max_theta: float = 0):
        r0_i = float(r0)
        if r0_i <= 0:
            raise ValueError("r0_layer must be > 0")

        extraN = int(math.ceil(self.N + 2*(float(max_theta) * float(altitude_m)) / self.dx))
        extraN = (extraN + 63) & ~63
        
        scr = InfiniteVonKarmanScreen2D(
            N=extraN, dx=self.dx, r0=r0_i, L0=float(L0),
            seed=self.seed + int(seed_offset),
            n_columns=self.n_columns,
            dtype=self.dtype,
        )

        layer_obj = _Layer(
            N= extraN,
            name=name,
            altitude_m=float(altitude_m),
            L0=float(L0),
            wind=(float(wind[0]), float(wind[1])),
            seed_offset=int(seed_offset),
            r0_layer=r0_i,
            screen=scr,
            phi=scr.scrn,
        )
        self._layers_obj.append(layer_obj)

        # dict view
        self.layers.append({
            "name": name,
            "h": layer_obj.altitude_m,
            "r0": layer_obj.r0_layer,
            "L0": layer_obj.L0,
            "vx": layer_obj.wind[0],
            "vy": layer_obj.wind[1],
            "active": True,
            "phi": layer_obj.phi,
        })


        self._refresh_totals()
        self._rebuild_name_map()

    # ----- updates -----
    def set_layer_active(self, idx: int, active: bool):
        idx = int(idx)
        a = bool(active)
        self._layers_obj[idx].active = a
        self.layers[idx]["active"] = a
        self._refresh_totals()  # if your total r0 depends on active layers

    def set_layer_active_name(self, name: str, active: bool):
        i = self._idx_from_name(name)
        self.set_layer_active(i, active)


    def set_layer_altitude(self, idx: int, altitude_m: float):
        idx = int(idx)
        h = float(altitude_m)

        # object model (used by sampling)
        self._layers_obj[idx].altitude_m = h

        # dict model (used by various GUI/worker code)
        self.layers[idx]["h"] = h

    def set_layer_wind(self, idx: int, vx: float, vy: float):
        L = self._layers_obj[int(idx)]
        L.wind = (float(vx), float(vy))
        self.layers[idx]["vx"] = float(vx)
        self.layers[idx]["vy"] = float(vy)

    def set_layer_r0(self, idx: int, r0_new: float):
        idx = int(idx)
        r0_new = float(r0_new)
        if r0_new <= 0:
            raise ValueError("r0_layer must be > 0")

        L = self._layers_obj[idx]
        L.r0_layer = r0_new
        L.screen.rescale_r0(r0_new)

        self.layers[idx]["r0"] = r0_new
        self._refresh_totals()

    def set_layer_L0(self, idx: int, L0_new: float):
        idx = int(idx)
        L = self._layers_obj[idx]
        L0_new = float(L0_new)
        if L0_new == L.L0:
            return
        L.L0 = L0_new
        L.screen.rebuild_L0(L0_new)
        self.layers[idx]["L0"] = L0_new

    # ----- time step -----
    def advance(self, dt: float):
        dt = float(dt)
        for i, L in enumerate(self._layers_obj):
            if not L.active:
                continue
            vx, vy = L.wind
            L.screen.advance(vx, vy, dt)
            L.phi = L.screen.scrn
            self.layers[i]["phi"] = L.phi


    # ----- batched sampling -----
    def sample_patches_batched(
        self,
        thetas_xy_rad,          # (S,2) radians
        size_pixels,            # patch side length in WORLD pixels
        M=128,
        angle_deg=0.0,
        ranges_m=None,          # (S,) meters; np.inf for NGS
        remove_piston=True,
        return_gpu=True,
        return_per_layer=False,
        layer_first=True,
    ):
        thetas = cp.asarray(thetas_xy_rad, dtype=self.dtype)
        if thetas.ndim != 2 or thetas.shape[1] != 2:
            raise ValueError("thetas_xy_rad must be shape (S,2)")
        S = int(thetas.shape[0])

        # Guide-star ranges (meters), used for LGS cone effect:
        #   alpha(h) = 1 - h / H
        if ranges_m is None:
            ranges = cp.full((S,), cp.inf, dtype=self.dtype)
        else:
            ranges = cp.asarray(ranges_m, dtype=self.dtype)
            if ranges.ndim != 1:
                ranges = ranges.reshape(-1)
            if ranges.size == 1:
                ranges = cp.full((S,), float(ranges.item()), dtype=self.dtype)
            elif ranges.size != S:
                raise ValueError(f"ranges_m must be scalar or shape (S,), got {tuple(ranges.shape)} for S={S}")

        side = float(size_pixels)
        M = int(M)

        # patch grid (relative coordinates centered at 0) in WORLD pixel units
        step = side / float(M)
        lin = (cp.arange(M, dtype=self.dtype) - (M - 1) / 2.0) * step
        Xs, Ys = cp.meshgrid(lin, lin, indexing="xy")

        # rotate the relative grid once
        a = math.radians(float(angle_deg))
        ca, sa = math.cos(a), math.sin(a)
        Xrel = ca * Xs - sa * Ys
        Yrel = sa * Xs + ca * Ys

        # broadcast relative grid to (S,M,M)
        Xrel_b = Xrel[None, :, :]
        Yrel_b = Yrel[None, :, :]

        if return_per_layer:
            L = len(self._layers_obj)
            outL = cp.zeros((L, S, M, M), dtype=self.dtype)

            for li, Lyr in enumerate(self._layers_obj):
                if not Lyr.active:
                    continue

                # --- AUTO CENTER from this layer's screen.N ---
                Nl = int(Lyr.screen.N)
                cx = (Nl - 1) / 2.0
                cy = (Nl - 1) / 2.0

                # base patch in WORLD coords for this layer
                cx0 = self.dtype(cx)
                cy0 = self.dtype(cy)
                Xbase = Xrel_b + cx0
                Ybase = Yrel_b + cy0

                h = float(Lyr.altitude_m)
                sx = (h * thetas[:, 0]) / self.dx
                sy = (h * thetas[:, 1]) / self.dx

                # LGS cone scaling (alpha=1 for NGS)
                alpha = 1.0 - (self.dtype(h) / ranges)
                alpha = cp.where(cp.isfinite(ranges), alpha, self.dtype(1.0))
                alpha = cp.clip(alpha, 0.0, 1.0)

                Xw = cx0 + alpha[:, None, None] * (Xbase - cx0) + sx[:, None, None]
                Yw = cy0 + alpha[:, None, None] * (Ybase - cy0) + sy[:, None, None]

                outL[li] = Lyr.screen.sample_bilinear_world(Xw, Yw, autopan=False, margin=2.0)

            if remove_piston:
                outL -= cp.mean(outL, axis=(2, 3), keepdims=True)

            if not layer_first:
                outL = cp.transpose(outL, (1, 0, 2, 3))  # (S,L,M,M)

            return outL if return_gpu else cp.asnumpy(outL)

        # combined (S,M,M)
        out = cp.zeros((S, M, M), dtype=self.dtype)

        for Lyr in self._layers_obj:
            if not Lyr.active:
                continue

            # --- AUTO CENTER from this layer's screen.N ---
            Nl = int(Lyr.screen.N)
            cx = (Nl - 1) / 2.0
            cy = (Nl - 1) / 2.0

            cx0 = self.dtype(cx)
            cy0 = self.dtype(cy)
            Xbase = Xrel_b + cx0
            Ybase = Yrel_b + cy0

            h = float(Lyr.altitude_m)
            sx = (h * thetas[:, 0]) / self.dx
            sy = (h * thetas[:, 1]) / self.dx

            # LGS cone scaling (alpha=1 for NGS)
            alpha = 1.0 - (self.dtype(h) / ranges)
            alpha = cp.where(cp.isfinite(ranges), alpha, self.dtype(1.0))
            alpha = cp.clip(alpha, 0.0, 1.0)

            Xw = cx0 + alpha[:, None, None] * (Xbase - cx0) + sx[:, None, None]
            Yw = cy0 + alpha[:, None, None] * (Ybase - cy0) + sy[:, None, None]

            out += Lyr.screen.sample_bilinear_world(Xw, Yw, autopan=True, margin=2.0)

        if remove_piston:
            out -= cp.mean(out, axis=(1, 2), keepdims=True)

        return out if return_gpu else cp.asnumpy(out)


    def sample_patches_batched_arcsec(self, thetas_xy_arcsec, *args, **kwargs):
        thetas = cp.asarray(thetas_xy_arcsec, dtype=self.dtype) * ARCSEC_TO_RAD
        return self.sample_patches_batched(thetas, *args, **kwargs)
    
    def get_layer_view(self, idx: int, size_pixels=None, M=None, angle_deg=0.0, return_gpu=True):
        idx = int(idx)

        if size_pixels is None:
            size_pixels = self.N
        if M is None:
            M = self.N

        patch = self._layers_obj[idx].screen.view_patch(
            size_pixels=size_pixels,
            M=M,
            angle_deg=angle_deg,
            margin=2.0,
            center_xy_pix=None,   # auto-center from that layer's N
        )
        return patch if return_gpu else cp.asnumpy(patch)


    def hard_reset(self):
        for i, L in enumerate(self._layers_obj):
            L.screen.hard_reset()
            L.phi = L.screen.scrn
            self.layers[i]["phi"] = L.phi
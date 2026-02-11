import math
import cupy as cp
import cupyx.scipy.ndimage as cndi
from typing import Sequence
from scripts.utilities import params, Pupil_tools


# -----------------------------
# Tip-tilt (LO) reconstructor
# -----------------------------
class TipTiltReconstructor_CuPy:
    """2-DOF tip/tilt reconstructor from Shack-Hartmann slopes.

    This is a dedicated *low-order* reconstructor that projects WFS slopes onto the
    global TT modes using weighted least squares (equivalently: weighted mean slopes).

    - Uses only NGS WFSs (non-LGS) when estimating TT.
    - Expresses the reconstructed TT in 500 nm reference units via per-sensor scaling (wl/500nm).
    - Provides an in-place TT stripping helper for LGS slopes (used before HO recon).
    """

    def __init__(self, sensors, wfs_is_lgs, wfs_wavelengths, weight_mode: str = "pupil"):
        self.sensors = sensors
        self.wfs_is_lgs = list(wfs_is_lgs) if wfs_is_lgs is not None else None
        self.wfs_wavelengths = list(wfs_wavelengths) if wfs_wavelengths is not None else None
        self.weight_mode = str(weight_mode).lower().strip()
        self.w_norm = []  # list of (n_active,) vectors (sum=1)
        self._build_weights()

    def _build_weights(self):
        self.w_norm = []
        if self.sensors is None:
            return

        for s in self.sensors:
            w = None
            if self.weight_mode == "pupil" and hasattr(s, "sub_pupils") and (getattr(s, "sub_pupils", None) is not None):
                try:
                    sp = cp.asarray(s.sub_pupils, dtype=cp.float32)
                    # mean pupil transmission per active subap
                    w = cp.mean(sp.reshape(sp.shape[0], -1), axis=1).astype(cp.float32, copy=False)
                except Exception:
                    w = None

            if w is None:
                try:
                    n = int(getattr(s, "active_sub_aps", 0))
                    if n <= 0 and hasattr(s, "sub_pupils") and (getattr(s, "sub_pupils", None) is not None):
                        n = int(cp.asarray(s.sub_pupils).shape[0])
                    if n <= 0:
                        n = 1
                    w = cp.ones((n,), dtype=cp.float32)
                except Exception:
                    w = cp.ones((1,), dtype=cp.float32)

            w = cp.asarray(w, dtype=cp.float32)
            ssum = cp.sum(w)
            # init-time sync is OK; ensures a robust fallback
            try:
                if float(ssum.get()) <= 0.0 or (not math.isfinite(float(ssum.get()))):
                    w = cp.ones_like(w)
                    ssum = cp.sum(w)
            except Exception:
                pass
            w = w / cp.maximum(ssum, cp.float32(1e-12))
            self.w_norm.append(w.astype(cp.float32, copy=False))

    def _weighted_mean_yx(self, slopes_yx: cp.ndarray, wi: int) -> cp.ndarray:
        """Return weighted mean (sy,sx) for one WFS slopes array."""
        s = cp.asarray(slopes_yx, dtype=cp.float32)
        if wi < len(self.w_norm):
            w = self.w_norm[wi]
        else:
            w = None

        if w is None or w.shape[0] != s.shape[0]:
            return cp.mean(s, axis=0).astype(cp.float32, copy=False)
        return cp.sum(s * w[:, None], axis=0).astype(cp.float32, copy=False)

    def estimate_tt_yx_500nm(self, slopes_err_stack: cp.ndarray) -> cp.ndarray:
        """Estimate global TT from NGS sensors. Returns [sy,sx] in 500nm-ref units."""
        if slopes_err_stack is None:
            return cp.zeros((2,), dtype=cp.float32)
        sl = cp.asarray(slopes_err_stack, dtype=cp.float32)
        n_wfs = int(sl.shape[0])

        if self.wfs_is_lgs is None or self.wfs_wavelengths is None:
            # fallback: treat all as NGS, no scaling
            acc = cp.zeros((2,), dtype=cp.float32)
            for wi in range(n_wfs):
                acc += self._weighted_mean_yx(sl[wi], wi)
            return acc / cp.float32(max(n_wfs, 1))

        ngs_idx = [i for i, is_lgs in enumerate(self.wfs_is_lgs[:n_wfs]) if not is_lgs]
        if not ngs_idx:
            return cp.zeros((2,), dtype=cp.float32)

        acc = cp.zeros((2,), dtype=cp.float32)
        for wi in ngs_idx:
            mu = self._weighted_mean_yx(sl[wi], wi)
            wl = cp.float32(self.wfs_wavelengths[wi])
            acc += mu * (wl / cp.float32(500e-9))
        return acc / cp.float32(len(ngs_idx))

    def strip_tt_inplace(self, slopes_err_stack: cp.ndarray, wfs_indices: Sequence[int]):
        """Subtract global TT (weighted mean slopes) from selected WFS slope arrays in-place."""
        if slopes_err_stack is None:
            return
        for wi in wfs_indices:
            if wi < 0 or wi >= int(slopes_err_stack.shape[0]):
                continue
            mu = self._weighted_mean_yx(slopes_err_stack[wi], wi)[None, :]
            slopes_err_stack[wi] = slopes_err_stack[wi] - mu



# -----------------------------
# Laplacian + waffle utilities
# -----------------------------
def build_grid_laplacian_and_ij_from_act_centers(act_centers_rc, grid_size, actuators):
    """
    Returns:
      G:  (n_act,n_act) unnormalized 4-neighbor graph Laplacian
      ij: (n_act,2) int32 lattice indices (i,j) in actuator grid coordinates
    """
    rc = cp.asarray(act_centers_rc).astype(cp.float32)
    n = int(rc.shape[0])

    step = float(grid_size) / float(actuators)
    ij = cp.rint((rc - (step / 2.0)) / step).astype(cp.int32)  # (n,2)

    # map lattice index -> compact actuator index (CPU dict is fine; n_act is modest)
    idx_of = {(int(i), int(j)): k for k, (i, j) in enumerate(ij)}

    G = cp.zeros((n, n), dtype=cp.float32)
    for k, (i, j) in enumerate(ij):
        deg = 0
        for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
            kk = idx_of.get((int(i + di), int(j + dj)))
            if kk is not None:
                G[k, kk] -= 1.0
                deg += 1
        G[k, k] += float(deg)

    return G.astype(cp.float32, copy=False), ij


def normalize_laplacian(G):
    d = cp.diag(G).astype(cp.float32)
    inv_sqrt_d = 1.0 / cp.sqrt(cp.maximum(d, cp.float32(1e-12)))
    Dm = cp.diag(inv_sqrt_d.astype(cp.float32))
    return (Dm @ G @ Dm).astype(cp.float32, copy=False)


def slopes_yx_to_vec(slopes_yx_cp):
    sl = cp.asarray(slopes_yx_cp)
    sy = sl[:, 0].astype(cp.float32, copy=False)
    sx = sl[:, 1].astype(cp.float32, copy=False)
    return cp.concatenate([sx, sy], axis=0)


def slopes_yx_batch_to_mat(slopes_batch_yx_cp):
    sl = cp.asarray(slopes_batch_yx_cp)
    sy = sl[..., 0].astype(cp.float32, copy=False)
    sx = sl[..., 1].astype(cp.float32, copy=False)
    return cp.concatenate([sx, sy], axis=1)


def _slopes_any_to_vec(one):
    a = cp.asarray(one)
    if a.ndim == 1:
        return a.astype(cp.float32, copy=False)
    if a.ndim == 2 and a.shape[1] == 2:
        return slopes_yx_to_vec(a)
    raise ValueError(f"Unsupported slopes shape {a.shape}; expected (nsub,2) or (2*nsub,)")


# -----------------------------
# Main reconstructor (with waffle constraint)
# -----------------------------
class TomoOnAxisIM_CuPy:
    def __init__(
        self,
        sensors,
        layer_heights_m,
        dx_world_m,
        M=None,
        size_pixels=None,
        angle_deg=0.0,
        reg_alpha=1e-2,
        reg_beta=0.0,
        reg_beta_normalize=True,
        reg_beta_scale_aware=True,
        map_pad_pixels=16,
        map_order=1,
        use_central_diff=True,
        remove_tt_in_output=True,
        chol_dtype=cp.float32,
        science_sensor_index=None,
        recon_sensor_indices=None,
        slopes_aggregation="stack",

        # ---- waffle control ----
        remove_waffle_in_solution=True,   # hard projection after solve
        waffle_per_layer=True,            # remove waffle independently per layer (recommended)
        reg_waffle_gamma=0.0,             # soft penalty strength (0 disables)
    ):
        self.sensors = list(sensors)
        self.layer_heights_m = [float(h) for h in layer_heights_m]
        self.dx_world_m = float(dx_world_m)

        self.M = int(M if M is not None else self.sensors[0].grid_size)
        self.size_pixels = float(size_pixels if size_pixels is not None else self.M)
        self.angle_deg = float(angle_deg)

        self.reg_alpha = float(reg_alpha)
        self.reg_beta = float(reg_beta)
        self.reg_beta_normalize = bool(reg_beta_normalize)
        self.reg_beta_scale_aware = bool(reg_beta_scale_aware)

        self.map_pad_pixels = int(map_pad_pixels)
        self.map_order = int(map_order)

        self.use_central_diff = bool(use_central_diff)
        self.remove_tt_in_output = bool(remove_tt_in_output)
        self.chol_dtype = chol_dtype

        self.slopes_aggregation = str(slopes_aggregation).lower()
        if self.slopes_aggregation not in ("stack", "mean"):
            raise ValueError("slopes_aggregation must be 'stack' or 'mean'")

        # waffle options
        self.remove_waffle_in_solution = bool(remove_waffle_in_solution)
        self.waffle_per_layer = bool(waffle_per_layer)
        self.reg_waffle_gamma = float(reg_waffle_gamma)

        # --- choose recon sensors ---
        if recon_sensor_indices is not None:
            idx = list(recon_sensor_indices)
        else:
            idx = list(range(len(self.sensors)))
            if science_sensor_index is not None:
                idx = [i for i in idx if i != int(science_sensor_index)]
        if len(idx) == 0:
            raise ValueError("No recon sensors selected.")

        self.recon_sensor_indices = idx
        self.recon_sensors = [self.sensors[i] for i in idx]
        self.science_sensor = None if science_sensor_index is None else self.sensors[int(science_sensor_index)]

        # pupil
        pupil_src = self.sensors[0] if self.science_sensor is None else self.science_sensor
        self.pupil = cp.asarray(pupil_src.pupil, dtype=cp.float32)

        # basis with consistent actuator ordering
        self.poke_amp = float(params.get("poke_amplitude"))
        self.actuators = int(params.get("actuators"))

        self.act_centers_rc = Pupil_tools.generate_actuators(
            pupil=self.pupil, actuators=self.actuators, grid_size=self.M
        )

        # IMPORTANT: pass act_centers so influence maps ordering matches
        basis_maps = Pupil_tools.generate_actuator_influence_map(
            act_centers=self.act_centers_rc,
            pupil=self.pupil,
            actuators=self.actuators,
            poke_amplitude=self.poke_amp,
            grid_size=self.M,
        )

        scale = cp.float32(4.0 * cp.pi / float(params.get("wfs_lambda")))
        self.basis = [cp.asarray(b, dtype=cp.float32) * scale for b in basis_maps]

        self.nL = len(self.layer_heights_m)
        self.nB = len(self.basis)
        self.nModes = self.nL * self.nB

        # ---- compatibility aliases (older/newer codepaths) ----
        # Some runtime code (e.g. scheduler/batch runner) expects snake_case names.
        # Keep both to avoid attribute errors.
        self.n_modes = self.nModes
        self.n_layers = self.nL
        self.n_basis = self.nB

        self.A = None
        self.At = None

        self._ref_stack = None
        self._ref_blocks = None
        self._ref_mean = None
        self._offsets = None

        # Laplacian + ij indices (for waffle)
        self.G = None
        self.Gn = None
        self._ij = None

        if (self.reg_beta > 0) or (self.remove_waffle_in_solution) or (self.reg_waffle_gamma > 0):
            G, ij = build_grid_laplacian_and_ij_from_act_centers(
                self.act_centers_rc, grid_size=self.M, actuators=self.actuators
            )
            self.G = G
            self._ij = ij
            if self.reg_beta > 0 and self.reg_beta_normalize:
                self.Gn = normalize_laplacian(self.G)

        # Build waffle vectors once (GPU)
        self._W_waffle = None  # (nModes, k) where k=1 (global) or k=nL (per-layer)
        self._init_waffle_modes()

        # cached eig solver parts
        self._eig_V = None
        self._eig_inv = None
        self._At = None

        # science-angle projection cache
        self._Xbase = None
        self._Ybase = None
        self._cx0 = None
        self._cy0 = None
        self._Xw_sci = None
        self._Yw_sci = None
        self._sci_valid = False

        # RT buffers
        self._rt_phase_acc = None
        self._rt_layer_pad = None

    # -----------------------------
    # Waffle setup and projection
    # -----------------------------
    def _init_waffle_modes(self):
        """
        Build waffle mode(s) in coefficient space matching basis ordering.
        Uses ij lattice indices derived from act_centers_rc mapping.
        """
        if self._ij is None:
            return

        ij = self._ij.astype(cp.int32, copy=False)  # (nB,2)
        # w[i] = (-1)^(i+j)
        parity = (ij[:, 0] + ij[:, 1]) & 1
        w1 = (1.0 - 2.0 * parity.astype(cp.float32))  # -> +1 on even, -1 on odd

        # normalize for numerical stability
        w1 = w1 / cp.sqrt(cp.maximum(cp.sum(w1 * w1), cp.float32(1e-12)))

        if not self.waffle_per_layer or self.nL == 1:
            # single global waffle vector over all modes: repeat w1 for each layer
            w = cp.concatenate([w1 for _ in range(self.nL)], axis=0)  # (nModes,)
            w = w / cp.sqrt(cp.maximum(cp.sum(w * w), cp.float32(1e-12)))
            self._W_waffle = w[:, None]  # (nModes,1)
        else:
            # independent waffle per layer: block-diagonal columns
            cols = []
            for li in range(self.nL):
                v = cp.zeros((self.nModes,), dtype=cp.float32)
                s = li * self.nB
                v[s:s + self.nB] = w1
                cols.append(v)
            W = cp.stack(cols, axis=1)  # (nModes, nL)
            # columns already normalized; ensure numeric safety
            self._W_waffle = W.astype(cp.float32, copy=False)

    def _project_out_waffle(self, x):
        """
        Hard projection: x <- x - W (W^T W)^-1 W^T x
        Here W has shape (nModes, k).
        """
        W = self._W_waffle
        if W is None:
            return x

        # For our construction, columns are orthonormal-ish; still do stable solve
        WT = W.T
        WT_W = WT @ W  # (k,k)
        rhs = WT @ x   # (k,)
        # solve (WT_W) a = rhs
        a = cp.linalg.solve(WT_W, rhs)
        return x - (W @ a)

    # -----------------------------
    # geometry helpers
    # -----------------------------
    def _precompute_base_grid(self, M, size_pixels, angle_deg):
        M = int(M)
        side = float(size_pixels)
        step = side / float(M)
        lin = (cp.arange(M, dtype=cp.float32) - (M - 1) / 2.0) * step
        Xs, Ys = cp.meshgrid(lin, lin, indexing="xy")

        a = math.radians(float(angle_deg))
        ca, sa = math.cos(a), math.sin(a)
        Xrel = ca * Xs - sa * Ys
        Yrel = sa * Xs + ca * Ys

        cx0 = cp.float32((M - 1) / 2.0)
        cy0 = cp.float32((M - 1) / 2.0)
        Xbase = Xrel + cx0
        Ybase = Yrel + cy0
        return Xbase, Ybase, cx0, cy0

    def _precompute_coords_per_layer_sensor(self, layer_heights_m, sensors, Xbase, Ybase, cx0, cy0, dx_world_m):
        nL = len(layer_heights_m)
        nS = len(sensors)
        Xw = [[None] * nS for _ in range(nL)]
        Yw = [[None] * nS for _ in range(nL)]

        for li, h in enumerate(layer_heights_m):
            h = float(h)
            for si, s in enumerate(sensors):
                thx, thy = float(s.dx), float(s.dy)
                H = float(s.gs_range_m) if math.isfinite(float(s.gs_range_m)) else float("inf")

                sx = cp.float32((h * thx) / float(dx_world_m))
                sy = cp.float32((h * thy) / float(dx_world_m))

                if math.isfinite(H):
                    alpha = cp.float32(1.0 - h / H)
                    alpha = cp.clip(alpha, 0.0, 1.0)
                else:
                    alpha = cp.float32(1.0)

                Xw[li][si] = cx0 + alpha * (Xbase - cx0) + sx
                Yw[li][si] = cy0 + alpha * (Ybase - cy0) + sy

        return Xw, Yw

    # -----------------------------
    # build interaction matrix
    # -----------------------------
    def build_interaction_matrix(self, chunk_modes=64, sensor_method="southwell"):
        sensors = self.recon_sensors
        nS = len(sensors)

        ref_blocks = []
        offsets = [0]
        for s in sensors:
            _, slopes, _ = s.measure(
                phase_map=cp.zeros((self.M, self.M), cp.float32),
                return_image=False,
                method=sensor_method
            )
            v = slopes_yx_to_vec(slopes[0])
            ref_blocks.append(v)
            offsets.append(offsets[-1] + int(v.size))

        if self.slopes_aggregation == "stack":
            ref_stack = cp.concatenate(ref_blocks, axis=0)
            n_slopes = int(ref_stack.size)
            self._ref_stack = ref_stack.astype(self.chol_dtype, copy=False)
            self._ref_blocks = [rb.astype(self.chol_dtype, copy=False) for rb in ref_blocks]
            self._offsets = offsets
        else:
            n0 = int(ref_blocks[0].size)
            if any(int(rb.size) != n0 for rb in ref_blocks):
                raise ValueError("mean aggregation requires identical slope vector sizes across recon sensors.")
            ref_mean = cp.mean(cp.stack(ref_blocks, axis=0), axis=0)
            n_slopes = int(ref_mean.size)
            self._ref_mean = ref_mean.astype(self.chol_dtype, copy=False)
            self._ref_blocks = [rb.astype(self.chol_dtype, copy=False) for rb in ref_blocks]
            self._offsets = offsets

        A = cp.empty((n_slopes, self.nModes), dtype=cp.float32)

        Xbase, Ybase, cx0, cy0 = self._precompute_base_grid(self.M, self.size_pixels, self.angle_deg)
        Xw, Yw = self._precompute_coords_per_layer_sensor(
            self.layer_heights_m, sensors, Xbase, Ybase, cx0, cy0, self.dx_world_m
        )

        pad = int(self.map_pad_pixels)
        order = int(self.map_order)

        col = 0
        total = self.nModes
        while col < total:
            cend = min(total, col + int(chunk_modes)) if (chunk_modes and chunk_modes > 0) else col + 1
            k = cend - col

            ph_p = [cp.empty((k, self.M, self.M), dtype=cp.float32) for _ in sensors]
            ph_m = [cp.empty((k, self.M, self.M), dtype=cp.float32) for _ in sensors] if self.use_central_diff else None

            for kk in range(k):
                mode = col + kk
                li = mode // self.nB
                bi = mode - li * self.nB

                delta = self.poke_amp * self.basis[bi]

                if pad > 0:
                    delta_samp = cp.pad(delta, pad, mode="constant", constant_values=0.0)
                else:
                    delta_samp = delta

                for si in range(nS):
                    yy = Yw[li][si] + (pad if pad > 0 else 0)
                    xx = Xw[li][si] + (pad if pad > 0 else 0)
                    mapped = cndi.map_coordinates(
                        delta_samp,
                        cp.asarray([yy, xx]),
                        order=order,
                        mode="constant",
                        cval=0.0
                    ).astype(cp.float32, copy=False)

                    ph_p[si][kk] = mapped
                    if self.use_central_diff:
                        ph_m[si][kk] = -mapped

            if self.slopes_aggregation == "stack":
                Sp = cp.empty((k, n_slopes), dtype=cp.float32)
                Sm = cp.empty((k, n_slopes), dtype=cp.float32) if self.use_central_diff else None

                for si, s in enumerate(sensors):
                    _, slp, _ = s.measure(
                        phase_map=ph_p[si],
                        return_image=False,
                        assume_radians=True,
                        method=sensor_method
                    )
                    vp = slopes_yx_batch_to_mat(slp)
                    vp = vp - self._ref_blocks[si][None, :].astype(cp.float32, copy=False)
                    Sp[:, offsets[si]:offsets[si + 1]] = vp

                    if self.use_central_diff:
                        _, slm, _ = s.measure(
                            phase_map=ph_m[si],
                            return_image=False,
                            assume_radians=True,
                            method=sensor_method
                        )
                        vm = slopes_yx_batch_to_mat(slm)
                        vm = vm - self._ref_blocks[si][None, :].astype(cp.float32, copy=False)
                        Sm[:, offsets[si]:offsets[si + 1]] = vm

                D = (Sp - Sm) / (2.0 * self.poke_amp) if self.use_central_diff else (Sp / self.poke_amp)
                A[:, col:cend] = D.T

            else:
                Spm = None
                Smm = None
                for si, s in enumerate(sensors):
                    _, slp, _ = s.measure(
                        phase_map=ph_p[si],
                        return_image=False,
                        assume_radians=True,
                        method=sensor_method
                    )
                    vp = slopes_yx_batch_to_mat(slp)
                    vp = vp - self._ref_blocks[si][None, :].astype(cp.float32, copy=False)
                    Spm = vp if Spm is None else (Spm + vp)

                    if self.use_central_diff:
                        _, slm, _ = s.measure(
                            phase_map=ph_m[si],
                            return_image=False,
                            assume_radians=True,
                            method=sensor_method
                        )
                        vm = slopes_yx_batch_to_mat(slm)
                        vm = vm - self._ref_blocks[si][None, :].astype(cp.float32, copy=False)
                        Smm = vm if Smm is None else (Smm + vm)

                Spm = Spm * (1.0 / float(nS))
                if self.use_central_diff:
                    Smm = Smm * (1.0 / float(nS))
                    D = (Spm - Smm) / (2.0 * self.poke_amp)
                else:
                    D = Spm / self.poke_amp
                A[:, col:cend] = D.T

            col = cend
            if not (chunk_modes and chunk_modes > 0):
                break

        self.A = A
        self.At = A.T
        cp.get_default_memory_pool().free_all_blocks()
        return A

    # -----------------------------
    # runtime prep
    # -----------------------------
    def prepare_runtime(self):
        M = self.M
        self._basis_flat = cp.stack([b.ravel() for b in self.basis], axis=0).astype(cp.float32, copy=False)

        pup = self.pupil.astype(cp.float32, copy=False)
        self._pupil_flat = pup.ravel()
        self._sel = self._pupil_flat > 0

        yy, xx = cp.indices((M, M), dtype=cp.float32)
        xx = (xx - (M - 1) * 0.5).ravel()[self._sel]
        yy = (yy - (M - 1) * 0.5).ravel()[self._sel]
        ones = cp.ones_like(xx)

        Xt = cp.stack([xx, yy, ones], axis=0)
        XtX = Xt @ Xt.T
        self._Xt = Xt
        self._XtX_inv = cp.linalg.inv(XtX)
        self._xx_sel = xx
        self._yy_sel = yy

        self._Xbase, self._Ybase, self._cx0, self._cy0 = self._precompute_base_grid(
            self.M, self.size_pixels, self.angle_deg
        )

        # RT buffers
        self._rt_phase_acc = cp.zeros((M, M), dtype=cp.float32)
        pad = int(self.map_pad_pixels)
        if pad > 0:
            Mp = M + 2 * pad
            self._rt_layer_pad = cp.zeros((Mp, Mp), dtype=cp.float32)
        else:
            self._rt_layer_pad = None

    # -----------------------------
    # regularization helpers
    # -----------------------------
    def _beta_eff(self):
        beta = float(self.reg_beta)
        if beta <= 0:
            return cp.float32(0.0)
        if not self.reg_beta_scale_aware:
            return cp.float32(beta)
        pitch_px = self.M / float(self.actuators)
        pitch_m = pitch_px * float(self.dx_world_m)
        return cp.float32(beta / (pitch_m * pitch_m))

    def _G_big(self):
        G_used = self.Gn if (self.Gn is not None) else self.G
        if G_used is None:
            return None
        if self.nL == 1:
            return G_used
        return cp.kron(cp.eye(self.nL, dtype=cp.float32), G_used)

    # -----------------------------
    # factorize (includes optional soft waffle penalty)
    # -----------------------------
    def factorize(self, rcond=1e-3):
        A = self.A.astype(cp.float32, copy=False)
        H = A.T @ A

        # Laplacian smoothing
        if (self.reg_beta > 0) and (self.G is not None):
            H = H + self._beta_eff() * self._G_big()

        # Soft waffle penalty (rank-k)
        if (self.reg_waffle_gamma > 0) and (self._W_waffle is not None):
            gamma = cp.float32(self.reg_waffle_gamma)
            W = self._W_waffle.astype(cp.float32, copy=False)  # (nModes,k)
            H = H + gamma * (W @ W.T)  # rank-k update

        # Ridge
        H = H + (self.reg_alpha * self.reg_alpha) * cp.eye(H.shape[0], dtype=H.dtype)

        evals, V = cp.linalg.eigh(H)
        idx = cp.argsort(evals)[::-1]
        evals = evals[idx]
        V = V[:, idx]

        e0 = evals[0]
        keep = evals >= (rcond * rcond * e0)

        self._eig_V = V[:, keep]
        self._eig_inv = (1.0 / evals[keep]).astype(cp.float32)
        self._At = A.T

    # -----------------------------
    # solve (with optional hard waffle projection)
    # -----------------------------
    def solve_coeffs(self, slopes_list_or_avg):
        if isinstance(slopes_list_or_avg, (list, tuple)):
            lst = list(slopes_list_or_avg)
        else:
            lst = [slopes_list_or_avg]

        if self.slopes_aggregation == "stack":
            if len(lst) != len(self.recon_sensors):
                raise ValueError(f"stack mode expects {len(self.recon_sensors)} slope blocks, got {len(lst)}")
            vecs = [_slopes_any_to_vec(sl) for sl in lst]
            s = cp.concatenate(vecs, axis=0).astype(self.chol_dtype, copy=False)
            if self._ref_stack is not None:
                s = s - self._ref_stack
        else:
            if len(lst) == 1:
                s = _slopes_any_to_vec(lst[0]).astype(self.chol_dtype, copy=False)
                if self._ref_mean is not None:
                    s = s - self._ref_mean
            else:
                if len(lst) != len(self.recon_sensors):
                    raise ValueError(f"mean mode expects 1 averaged slope block or {len(self.recon_sensors)} blocks.")
                acc = None
                for si, sl in enumerate(lst):
                    v = _slopes_any_to_vec(sl).astype(self.chol_dtype, copy=False)
                    v = v - self._ref_blocks[si]
                    acc = v if acc is None else (acc + v)
                s = acc * (1.0 / float(len(self.recon_sensors)))

        rhs = self._At @ s
        t = self._eig_V.T @ rhs
        t = t * self._eig_inv
        x = self._eig_V @ t

        # Hard waffle removal (projects out null/near-null checkerboard)
        if self.remove_waffle_in_solution and (self._W_waffle is not None):
            x = self._project_out_waffle(x.astype(cp.float32, copy=False))

        return x.astype(cp.float32, copy=False)

    # -----------------------------
    # fast solve for already-stacked slopes (stack aggregation)
    # -----------------------------
    def solve_coeffs_stack(self, slopes_stack):
        """Solve coefficients when slopes are already stacked into one array.

        Expected input shapes:
          - (nS, n_sub_active, 2)  (preferred)
          - (nS, 2*n_sub_active)
          - (n_slopes,) slope vector

        This avoids Python list handling + per-frame cp.concatenate.
        """
        if self.slopes_aggregation != "stack":
            # preserve behaviour
            return self.solve_coeffs(slopes_stack)

        s = cp.asarray(slopes_stack)
        if s.ndim == 3:
            # (nS, n_sub, 2) -> (n_slopes,)
            s = s.reshape(-1)
        elif s.ndim == 2:
            s = s.reshape(-1)
        elif s.ndim == 1:
            pass
        else:
            raise ValueError(f"solve_coeffs_stack expects 1D/2D/3D input, got shape {tuple(s.shape)}")

        s = s.astype(self.chol_dtype, copy=False)
        if self._ref_stack is not None:
            s = s - self._ref_stack

        rhs = self._At @ s
        t = self._eig_V.T @ rhs
        t = t * self._eig_inv
        x = self._eig_V @ t

        if self.remove_waffle_in_solution and (self._W_waffle is not None):
            x = self._project_out_waffle(x.astype(cp.float32, copy=False))

        return x.astype(cp.float32, copy=False)

    def reconstruct_onaxis_stack(self, slopes_stack, return_coeffs=False):
        """On-axis phase reconstruction for already-stacked slopes (stack aggregation)."""
        xhat = self.solve_coeffs_stack(slopes_stack)
        ph = self.coeffs_to_onaxis_phase(xhat)
        return (ph, xhat) if return_coeffs else ph

    # -----------------------------
    # science-angle cache + RT projection
    # -----------------------------
    def update_science_angle(self, theta_x_rad, theta_y_rad, range_m=float("inf")):
        if self._Xbase is None:
            raise RuntimeError("Call prepare_runtime() before update_science_angle().")

        thx = float(theta_x_rad)
        thy = float(theta_y_rad)
        H = float(range_m) if math.isfinite(float(range_m)) else float("inf")

        Xw_list = []
        Yw_list = []
        for h in self.layer_heights_m:
            h = float(h)
            sx = cp.float32((h * thx) / float(self.dx_world_m))
            sy = cp.float32((h * thy) / float(self.dx_world_m))

            if math.isfinite(H):
                alpha = cp.float32(1.0 - h / H)
                alpha = cp.clip(alpha, 0.0, 1.0)
            else:
                alpha = cp.float32(1.0)

            Xw_list.append(self._cx0 + alpha * (self._Xbase - self._cx0) + sx)
            Yw_list.append(self._cy0 + alpha * (self._Ybase - self._cy0) + sy)

        self._Xw_sci = Xw_list
        self._Yw_sci = Yw_list
        self._sci_valid = True

    def coeffs_to_science_phase_cached(self, xhat):
        if not self._sci_valid:
            raise RuntimeError("Call update_science_angle() before coeffs_to_science_phase_cached().")

        x = cp.asarray(xhat, dtype=cp.float32).reshape(self.nL, self.nB)

        pad = int(self.map_pad_pixels)
        order = int(self.map_order)

        ph_acc = self._rt_phase_acc
        ph_acc.fill(0.0)

        for li in range(self.nL):
            layer_flat = x[li] @ self._basis_flat
            layer_map = layer_flat.reshape(self.M, self.M).astype(cp.float32, copy=False)

            Xw = self._Xw_sci[li]
            Yw = self._Yw_sci[li]

            if pad > 0:
                lp = self._rt_layer_pad
                lp.fill(0.0)
                lp[pad:pad + self.M, pad:pad + self.M] = layer_map
                samp = cndi.map_coordinates(
                    lp,
                    cp.asarray([Yw + pad, Xw + pad]),
                    order=order,
                    mode="constant",
                    cval=0.0
                )
            else:
                samp = cndi.map_coordinates(
                    layer_map,
                    cp.asarray([Yw, Xw]),
                    order=order,
                    mode="constant",
                    cval=0.0
                )

            ph_acc += samp.astype(cp.float32, copy=False)

        # apply pupil + piston + optional TT removal
        ph_flat = ph_acc.ravel() * self._pupil_flat

        sel = self._sel
        z = ph_flat[sel]
        z = z - cp.mean(z)

        if self.remove_tt_in_output:
            beta = self._XtX_inv @ (self._Xt @ z)
            z = z - (beta[0] * self._xx_sel + beta[1] * self._yy_sel + beta[2])

        ph_flat = ph_flat.copy()
        ph_flat[sel] = z
        return ph_flat.reshape(self.M, self.M).astype(cp.float32, copy=False)

    def reconstruct_science_rt(self, slopes_list, theta_x_rad, theta_y_rad, range_m=float("inf"),
                               return_coeffs=False):
        self.update_science_angle(theta_x_rad, theta_y_rad, range_m=range_m)
        xhat = self.solve_coeffs(slopes_list)
        ph = self.coeffs_to_science_phase_cached(xhat)
        return (ph, xhat) if return_coeffs else ph

    # -----------------------------
    # keep on-axis output available
    # -----------------------------
    def coeffs_to_onaxis_phase(self, xhat):
        x = cp.asarray(xhat, dtype=cp.float32)
        coeff_sum = cp.sum(x.reshape(self.nL, self.nB), axis=0)
        phase_flat = coeff_sum @ self._basis_flat
        phase_flat = phase_flat * self._pupil_flat

        sel = self._sel
        z = phase_flat[sel]
        z = z - cp.mean(z)

        if self.remove_tt_in_output:
            beta = self._XtX_inv @ (self._Xt @ z)
            z = z - (beta[0] * self._xx_sel + beta[1] * self._yy_sel + beta[2])

        phase_flat = phase_flat.copy()
        phase_flat[sel] = z
        return phase_flat.reshape(self.M, self.M).astype(cp.float32, copy=False)

    def reconstruct_onaxis(self, slopes_list, return_coeffs=False):
        xhat = self.solve_coeffs(slopes_list)
        ph = self.coeffs_to_onaxis_phase(xhat)
        return (ph, xhat) if return_coeffs else ph

import math
import cupy as cp
import cupyx.scipy.ndimage as cndi
from scripts.utilities import params, Pupil_tools  # your utilities.py

try:
    from cupyx.scipy.linalg import solve_triangular as _solve_tri
except Exception:
    _solve_tri = None


def slopes_yx_to_vec(slopes_yx_cp):
    # slopes are (cy,cx) per subap -> stack as [sx, sy]
    sl = cp.asarray(slopes_yx_cp)
    sy = sl[:, 0].astype(cp.float32, copy=False)
    sx = sl[:, 1].astype(cp.float32, copy=False)
    return cp.concatenate([sx, sy], axis=0)


def project_like_sample_patches(
    src_patch,            # (M,M) cp.float32  (this is a "layer screen patch")
    h_m: float,
    theta_x_rad: float,
    theta_y_rad: float,
    range_m: float,       # inf for NGS; finite for LGS
    dx_world_m: float,    # EXACT same self.dx used in your sample_patches_batched() shifts
    size_pixels: float,   # patch side length in WORLD pixels
    M: int,
    angle_deg: float = 0.0,
):
    """
    Same mapping as sample_patches_batched(), but sampling from src_patch directly.
    Recommended: size_pixels == M (WORLD pixels match array pixels).
    """
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

    sx = cp.float32((float(h_m) * float(theta_x_rad)) / float(dx_world_m))
    sy = cp.float32((float(h_m) * float(theta_y_rad)) / float(dx_world_m))

    if math.isfinite(float(range_m)):
        alpha = cp.float32(1.0 - float(h_m) / float(range_m))
        alpha = cp.clip(alpha, 0.0, 1.0)
    else:
        alpha = cp.float32(1.0)

    Xw = cx0 + alpha * (Xbase - cx0) + sx
    Yw = cy0 + alpha * (Ybase - cy0) + sy

    return cndi.map_coordinates(src_patch, cp.asarray([Yw, Xw]), order=1, mode="constant", cval=0.0).astype(cp.float32, copy=False)


def remove_piston_and_tt(phase, pupil_mask, remove_tt=True):
    ph = phase.astype(cp.float32, copy=True)
    m = pupil_mask.astype(cp.float32, copy=False)

    denom = cp.sum(m) + 1e-12
    ph = ph - (cp.sum(ph * m) / denom) * m

    if not remove_tt:
        return ph

    N = ph.shape[0]
    yy, xx = cp.indices((N, N), dtype=cp.float32)
    xx = xx - (N - 1) * 0.5
    yy = yy - (N - 1) * 0.5
    sel = (m > 0)

    X = cp.stack([xx[sel].ravel(), yy[sel].ravel(), cp.ones(int(cp.sum(sel)), dtype=cp.float32)], axis=1)
    z = ph[sel].ravel()
    beta, *_ = cp.linalg.lstsq(X, z, rcond=None)
    plane = (beta[0] * xx + beta[1] * yy + beta[2]) * m
    return ph - plane


class TomoOnAxisIM_CuPy:
    """
    GPU-only tomographic IM reconstructor -> ON-AXIS phase.

    Basis is taken from utilities.Pupil_tools.generate_actuator_influence_map(), i.e.
    normalized by poke_amplitude and pupil-masked.
    """
    def __init__(
        self,
        sensors,
        layer_heights_m,
        dx_world_m,
        M=None,
        size_pixels=None,
        angle_deg=0.0,
        reg_alpha=1e-2,
        use_central_diff=True,
        remove_tt_in_output=True,
        chol_dtype=cp.float32,
    ):
        self.sensors = list(sensors)
        self.layer_heights_m = [float(h) for h in layer_heights_m]
        self.dx_world_m = float(dx_world_m)

        self.M = int(M if M is not None else self.sensors[0].grid_size)
        self.size_pixels = float(size_pixels if size_pixels is not None else self.M)
        self.angle_deg = float(angle_deg)

        self.reg_alpha = float(reg_alpha)
        self.use_central_diff = bool(use_central_diff)
        self.remove_tt_in_output = bool(remove_tt_in_output)
        self.chol_dtype = chol_dtype

        pupil = self.sensors[0].pupil
        self.pupil = cp.asarray(pupil, dtype=cp.float32)
        poke_amplitude = params.get("poke_amplitude")
        actuators = params.get("actuators")
        self.poke_amp = float(poke_amplitude)


        basis_maps = Pupil_tools.generate_actuator_influence_map(
            act_centers=None,
            pupil=self.pupil,
            actuators=actuators,
            poke_amplitude=self.poke_amp,
            grid_size=self.M,
            )
        self.basis = [cp.asarray(b, dtype=cp.float32) for b in basis_maps]

        self.nL = len(self.layer_heights_m)
        self.nB = len(self.basis)
        self.nModes = self.nL * self.nB

        self.A = None
        self.At = None
        self.L = None

    def _poke_phase_for_sensor(self, li, bi, sensor, sign=+1.0):
        # influence maps are normalized by poke_amp, so poke_amp * basis gives meters (surface/OPD-like)
        delta = (sign * self.poke_amp) * self.basis[bi]
        h = self.layer_heights_m[li]
        thx, thy = float(sensor.dx), float(sensor.dy)
        H = float(sensor.gs_range_m) if math.isfinite(float(sensor.gs_range_m)) else float("inf")

        return project_like_sample_patches(
            delta,
            h_m=h,
            theta_x_rad=thx,
            theta_y_rad=thy,
            range_m=H,
            dx_world_m=self.dx_world_m,
            size_pixels=self.size_pixels,
            M=self.M,
            angle_deg=self.angle_deg,
        )

    def _precompute_base_grid(self, M, size_pixels, angle_deg):
        """Precompute rotated (Xbase,Ybase) in patch-index coords."""
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
        """
        Precompute sampling coordinate maps (Xw,Yw) for each (layer,sensor).
        Returns lists Xw[li][si], Yw[li][si] (each (M,M) cp.float32).
        """
        nL = len(layer_heights_m)
        nS = len(sensors)
        Xw = [[None]*nS for _ in range(nL)]
        Yw = [[None]*nS for _ in range(nL)]

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

    def build_interaction_matrix(self, chunk_modes=128): # 128 chunk is roughly 4GB of vram

        # --- reference slopes per sensor---
        ref_blocks = []
        offsets = [0]
        for s in self.sensors:
            _, slopes, _ = s.measure(phase_map=cp.zeros((self.M, self.M), cp.float32), return_image=False)
            v = slopes_yx_to_vec(slopes[0])
            ref_blocks.append(v)
            offsets.append(offsets[-1] + int(v.size))
        ref = cp.concatenate(ref_blocks, axis=0)
        n_slopes = int(ref.size)
        self._offsets = offsets  # keep if you want

        A = cp.empty((n_slopes, self.nModes), dtype=cp.float32)

        # --- precompute geometry once ---
        Xbase, Ybase, cx0, cy0 = self._precompute_base_grid(self.M, self.size_pixels, self.angle_deg)
        Xw, Yw = self._precompute_coords_per_layer_sensor(
            self.layer_heights_m, self.sensors, Xbase, Ybase, cx0, cy0, self.dx_world_m
        )

        # reusable slope buffers to avoid concat
        sp = cp.empty((n_slopes,), dtype=cp.float32)
        sm = cp.empty((n_slopes,), dtype=cp.float32) if self.use_central_diff else None

        # measure all sensors for one set of per-sensor phase maps
        def measure_stack_into(buf, phase_maps_per_sensor):
            for si, s in enumerate(self.sensors):
                _, slopes, _ = s.measure(phase_map=phase_maps_per_sensor[si], return_image=False)
                v = slopes_yx_to_vec(slopes[0])  # (2*nsub,)
                buf[offsets[si]:offsets[si+1]] = v

        if chunk_modes and chunk_modes > 0:
            # Batch per sensor: build phase_map stack (n_map, M, M) then measure once per sensor per chunk
            col = 0
            total = self.nModes
            while col < total:
                cend = min(total, col + int(chunk_modes))
                k = cend - col

                # build per-sensor phase stacks for + and - (if central diff)
                ph_p = [cp.empty((k, self.M, self.M), dtype=cp.float32) for _ in self.sensors]
                ph_m = [cp.empty((k, self.M, self.M), dtype=cp.float32) for _ in self.sensors] if self.use_central_diff else None

                # fill chunk
                for kk in range(k):
                    mode = col + kk
                    li = mode // self.nB
                    bi = mode - li * self.nB
                    delta = self.poke_amp * self.basis[bi]  # (M,M), basis is normalized already

                    for si in range(len(self.sensors)):
                        ph_p[si][kk] = cndi.map_coordinates(delta, cp.asarray([Yw[li][si], Xw[li][si]]), order=1, mode="constant", cval=0.0)
                        if self.use_central_diff:
                            ph_m[si][kk] = -ph_p[si][kk]  # since mapping is linear, -delta maps to -phase

                # measure each sensor once for this chunk
                # store slopes for all k maps, then write into A columns
                # We do per-sensor and then interleave into full slope vector.
                # Build stacked slope arrays:
                slopes_p = []
                slopes_m = [] if self.use_central_diff else None

                for si, s in enumerate(self.sensors):
                    _, slp, _ = s.measure(phase_map=ph_p[si], return_image=False)  # slopes: (k, nsub, 2)
                    # convert each map to vector and stack (k, n_slope_si)
                    vp = cp.stack([slopes_yx_to_vec(slp[i]) for i in range(k)], axis=0)
                    slopes_p.append(vp)

                    if self.use_central_diff:
                        _, slm, _ = s.measure(phase_map=ph_m[si], return_image=False)
                        vm = cp.stack([slopes_yx_to_vec(slm[i]) for i in range(k)], axis=0)
                        slopes_m.append(vm)

                # assemble full stacked slope vectors for each kk into A
                # (k, n_slopes)
                Sp = cp.empty((k, n_slopes), dtype=cp.float32)
                Sm = cp.empty((k, n_slopes), dtype=cp.float32) if self.use_central_diff else None

                for si in range(len(self.sensors)):
                    Sp[:, offsets[si]:offsets[si+1]] = slopes_p[si]
                    if self.use_central_diff:
                        Sm[:, offsets[si]:offsets[si+1]] = slopes_m[si]

                Sp = Sp - ref[None, :]
                if self.use_central_diff:
                    Sm = Sm - ref[None, :]
                    D = (Sp - Sm) / (2.0 * self.poke_amp)
                else:
                    D = Sp / self.poke_amp

                # write columns
                A[:, col:cend] = D.T
                col = cend

        else:
            # no batching
            col = 0
            for li in range(self.nL):
                for bi in range(self.nB):
                    delta = self.poke_amp * self.basis[bi]  # (M,M)

                    # per-sensor phase maps 
                    ph_p = []
                    ph_m = [] if self.use_central_diff else None
                    for si, s in enumerate(self.sensors):
                        ph = cndi.map_coordinates(delta, cp.asarray([Yw[li][si], Xw[li][si]]), order=1, mode="constant", cval=0.0)
                        ph_p.append(ph)
                        if self.use_central_diff:
                            ph_m.append(-ph)

                    # measure and fill buffers
                    measure_stack_into(sp, ph_p)
                    sp = sp - ref

                    if self.use_central_diff:
                        measure_stack_into(sm, ph_m)
                        sm = sm - ref
                        d = (sp - sm) / (2.0 * self.poke_amp)
                    else:
                        d = sp / self.poke_amp

                    A[:, col] = d
                    col += 1

        self.A = A
        self.At = A.T

        cp.get_default_memory_pool().free_all_blocks()

        return A

    def prepare_runtime(self):
        M = self.M

        # (nB, P) flattened basis for fast matmul
        self._basis_flat = cp.stack([b.ravel() for b in self.basis], axis=0).astype(cp.float32, copy=False)

        # pupil selection (use same mask each time)
        pup = self.pupil.astype(cp.float32, copy=False)
        self._pupil_flat = pup.ravel()
        self._sel = self._pupil_flat > 0

        # coords (centered) only on pupil support
        yy, xx = cp.indices((M, M), dtype=cp.float32)
        xx = (xx - (M - 1) * 0.5).ravel()[self._sel]
        yy = (yy - (M - 1) * 0.5).ravel()[self._sel]
        ones = cp.ones_like(xx)

        # Precompute plane-fit operator: beta = (X^T X)^-1 X^T z
        # X: (Nsel,3) with columns [x,y,1]
        # Store Xt (3,Nsel) and inv(XtX) (3,3)
        Xt = cp.stack([xx, yy, ones], axis=0)  # (3, Nsel)
        XtX = Xt @ Xt.T                         # (3,3)
        self._Xt = Xt
        self._XtX_inv = cp.linalg.inv(XtX)
        self._xx_sel = xx
        self._yy_sel = yy


    def factorize(self):
        if self.A is None:
            raise RuntimeError("Call build_interaction_matrix() first.")
        A = self.A.astype(self.chol_dtype, copy=False)
        H = A.T @ A
        H = H + (self.reg_alpha * cp.eye(self.nModes, dtype=self.chol_dtype))
        self.L = cp.linalg.cholesky(H)
        self.At = A.T

    def solve_coeffs(self, slopes_list):
        if self.L is None:
            raise RuntimeError("Call factorize() first.")
        s = cp.concatenate([slopes_yx_to_vec(sl) for sl in slopes_list], axis=0).astype(self.chol_dtype, copy=False)
        rhs = self.At @ s

        if _solve_tri is not None:
            y = _solve_tri(self.L, rhs, lower=True)
            x = _solve_tri(self.L.T, y, lower=False)
        else:
            y = cp.linalg.solve(self.L, rhs)
            x = cp.linalg.solve(self.L.T, y)

        return x.astype(cp.float32, copy=False)

    def coeffs_to_onaxis_phase(self, xhat):
        # xhat: (nL*nB,)
        x = cp.asarray(xhat, dtype=cp.float32)

        # On-axis: sum over layers first 
        coeff_sum = cp.sum(x.reshape(self.nL, self.nB), axis=0)     # (nB,)

        # phase_flat = Σ_b coeff_sum[b] * basis[b,:]
        phase_flat = coeff_sum @ self._basis_flat                   # (P,)

        # apply pupil
        phase_flat = phase_flat * self._pupil_flat

        # piston remove 
        sel = self._sel
        z = phase_flat[sel]
        z = z - cp.mean(z)

        if self.remove_tt_in_output:
            # beta = inv(XtX) @ (Xt @ z)
            beta = self._XtX_inv @ (self._Xt @ z)  # (3,)
            z = z - (beta[0] * self._xx_sel + beta[1] * self._yy_sel + beta[2])

        # write back
        phase_flat = phase_flat.copy()
        phase_flat[sel] = z

        return phase_flat.reshape(self.M, self.M).astype(cp.float32, copy=False)


    def reconstruct_onaxis(self, slopes_list, return_coeffs=False):
        xhat = self.solve_coeffs(slopes_list)
        ph = self.coeffs_to_onaxis_phase(xhat)
        return (ph, xhat) if return_coeffs else ph

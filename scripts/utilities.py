import aotools
import cupy as cp
import numpy as np
#import scripts.AOtools_cuda.AOtools_infinitephasescreen_cuda as aot_cp

from pathlib import Path
import json

with open(Path(__file__).parent.parent / "config_default.json", "r") as f:
    config = json.load(f)

params = config.copy()

def set_params(new_params):
    global params
    params = new_params

class PhaseMap_tools:

    @staticmethod
    def generate_phase_map(grid_size=None, telescope_diameter=None, r0=None, L0=None, Vwind=None, px_scale=None, random_seed=None, **kwargs):
        # read current params when values not provided
        if grid_size is None:
            grid_size = params.get("grid_size")
        if telescope_diameter is None:
            telescope_diameter = params.get("telescope_diameter")
        if r0 is None:
            r0 = params.get("r0")
        if L0 is None:
            L0 = params.get("L0")
        if random_seed is None:
            random_seed = params.get("random_seed")
        if px_scale is None:
            px_scale = telescope_diameter/grid_size
        # create phase screen
        phase_screen = aotools.infinitephasescreen.PhaseScreenKolmogorov(int(cp.ceil(grid_size)), px_scale, r0, L0, random_seed=random_seed)
        phase_map = phase_screen.add_row

        # generate next rows based on wind vel and dt
        def advance_phase_map(frame_rate=500):
            shift_m = Vwind * 1/frame_rate
            shift_pix = shift_m / (telescope_diameter/grid_size)
            n_rows = int(np.round(abs(shift_pix)))
            for _ in range(max(1, n_rows-1)):
                phase_map()
            return phase_map()
        
        return phase_screen, advance_phase_map
    
    # generate influence map for an amplitude and spread factor sigma
    @staticmethod
    def gaussian(r0, c0, amp=1e-6, sigma=None, grid_size=None):
        if grid_size is None:
            grid_size = params.get("grid_size")
        if sigma is None:
            sigma = max(1.0, grid_size * 0.05)
        
        y, x = cp.meshgrid(cp.arange(grid_size), cp.arange(grid_size), indexing='ij')
        surf = amp * cp.exp(-((x - c0)**2 + (y - r0)**2) / (2 * sigma**2))
        return surf







class Pupil_tools:
    @staticmethod
    def circle(radius, grid_size, center=None):
            if center is None:
                center = (grid_size / 2, grid_size / 2)

            y = cp.arange(grid_size)[:, None]
            x = cp.arange(grid_size)[None, :]
            dist2 = (y - center[0])**2 + (x - center[1])**2
            mask = (dist2 <= radius**2).astype(cp.float32)
            return mask

    @staticmethod
    def generate_pupil(grid_size=None, telescope_center_obscuration=None):
        if grid_size is None:
            grid_size = params.get("grid_size")
        if telescope_center_obscuration is None:
            telescope_center_obscuration = params.get("telescope_center_obscuration")

        outer = Pupil_tools.circle(grid_size/2, grid_size)
        inner = Pupil_tools.circle(grid_size/2 * telescope_center_obscuration, grid_size)
        return cp.asarray(outer - inner)

    # generate actuator positions on pupil
    @staticmethod
    def generate_actuators(pupil=None, actuators=None, grid_size=None):
        if grid_size is None:
            grid_size = params.get("grid_size")
        if actuators is None:
            actuators = params.get("actuators")
        if pupil is None:
            pupil = Pupil_tools.generate_pupil(grid_size=grid_size)

        # create grid of actuator candidate centers
        step = grid_size / actuators
        grid_coords = cp.arange(step/2, grid_size, step)
        rr, cc = cp.meshgrid(grid_coords, grid_coords, indexing='ij')

        # round to nearest integer for indexing
        rr_idx = rr.astype(cp.int32)
        cc_idx = cc.astype(cp.int32)

        # keep only points inside the pupil
        mask = pupil[rr_idx, cc_idx] > 0
        act_centers = cp.stack([rr_idx[mask], cc_idx[mask]], axis=1)
        return act_centers

    # generate list of images that denote each actuator's influence (from 0 to 1) on the mirror
    @staticmethod
    def generate_actuator_influence_map(act_centers=None, pupil=None, actuators=None, poke_amplitude=None, grid_size=None):
        if pupil is None:
            pupil = Pupil_tools.generate_pupil(grid_size=grid_size)
        if grid_size is None:
            grid_size = params.get("grid_size")
        if actuators is None:
            actuators = params.get("actuators")
        if poke_amplitude is None:
            poke_amplitude = params.get("poke_amplitude")
        if act_centers is None:
            act_centers = Pupil_tools.generate_actuators(pupil=pupil, actuators=actuators, grid_size=grid_size)

        # Vectorized gaussian over actuators 
        sigma = grid_size / actuators * 0.6

        # create coordinate grids
        y, x = cp.meshgrid(cp.arange(grid_size), cp.arange(grid_size), indexing='ij')
        y = y[None, :, :]  # shape (1, grid_size, grid_size)
        x = x[None, :, :]

        # actuator centers
        r0 = act_centers[:, 0][:, None, None]  # (n_act, 1, 1)
        c0 = act_centers[:, 1][:, None, None]

        # Gaussian surfaces
        surf = cp.exp(-((x - c0)**2 + (y - r0)**2) / (2 * sigma**2))

        # normalize by poke amplitude and multiply by pupil
        influence_maps = (surf / surf.max(axis=(1,2), keepdims=True)) * pupil[None, :, :]  

        # scale to poke amplitude
        influence_maps *= poke_amplitude

        return influence_maps  # shape (n_act, grid_size, grid_size)



class Analysis:
    
    
    # create PSF given pupil and phase map
    @staticmethod
    def generate_science_image(pupil=None, phase_map=None, science_lambda=None, pad=None):
        if pad is None:
            pad = params.get("field_padding")
        if science_lambda is None:
            science_lambda = params.get("science_lambda")
        if pupil is None:
            pupil = Pupil_tools.generate_pupil()
        if phase_map is None:
            phase_map = cp.zeros_like(pupil)

        field = pupil * cp.exp(1j * phase_map)
        field_p = cp.pad(field,((pad,pad) , (pad,pad)), mode='constant')

        F = cp.fft.fftshift(cp.fft.fft2(field_p))
        I = cp.abs(F)**2

        F0 = cp.fft.fftshift(cp.fft.fft2(pupil))
        I0 = cp.abs(F0)**2

        return I,  I.max()/I0.max()
    
    @staticmethod
    def fwhm_radial(I):

        xp = cp.get_array_module(I)

        ny, nx = I.shape
        cy, cx = ny // 2, nx // 2

        y, x = xp.indices(I.shape)
        r = xp.sqrt((x - cx)**2 + (y - cy)**2)

        r_flat = r.ravel()
        I_flat = I.ravel()

        # radial bins (integer pixel radii)
        r_int = r_flat.astype(xp.int32)
        max_r = r_int.max() + 1

        # azimuthal average
        radial_sum = xp.bincount(r_int, weights=I_flat, minlength=max_r)
        radial_cnt = xp.bincount(r_int, minlength=max_r)
        radial_prof = radial_sum / radial_cnt

        radial_prof /= radial_prof[0]  # normalize to peak

        # find half-maximum
        half = 0.5
        idx = xp.where(radial_prof <= half)[0][0]

        # linear interpolation for sub-pixel accuracy
        r1, r2 = idx - 1, idx
        y1, y2 = radial_prof[r1], radial_prof[r2]
        r_half = r1 + (half - y1) / (y2 - y1)

        return 2.0 * r_half


import threading
gpu_lock = threading.Lock()

class WFSensor_tools:
    ARCSEC2RAD = cp.pi / (180.0 * 3600.0)
    RAD2ARCSEC = (180.0 * 3600.0) / cp.pi


    class ShackHartmann:
        def __init__(self, n_sub = None, wavelength = None, dx = None, dy = None, noise=0.0, pupil=None, grid_size=None):
            if n_sub is None:
                n_sub = params.get("sub_apertures")
            if wavelength is None:
                wavelength = params.get("wfs_lambda")
            if grid_size is None:
                grid_size = params.get("grid_size")
            if pupil is None:
                pupil = Pupil_tools.generate_pupil(grid_size=grid_size)
            if dx is None:
                dx = 0.0
            if dy is None:
                dy = 0.0

            self.n_sub = n_sub
            self.wavelength = wavelength
            self.noise = noise
            self.pupil = pupil
            self.grid_size = grid_size

            self.dx = dx * WFSensor_tools.ARCSEC2RAD
            self.dy = dy * WFSensor_tools.ARCSEC2RAD
            self.field_angle = (self.dx, self.dy)
            self.sen_angle = cp.sqrt(self.field_angle[0]**2 + self.field_angle[1]**2)

            self.active_sub_aps, self.sub_aps, self.sub_aps_idx, self.sub_ap_width, self.sub_slice, self.sub_pupils = self.generate_sub_apertures(pupil, grid_size)
            
        def recompute(self, n_sub=None, wavelength=None,  dx = None, dy = None, noise=0.0, pupil=None, grid_size=None):
            if n_sub is not None:
                self.n_sub = n_sub
            if wavelength is not None:
                self.wavelength = wavelength
            if noise is not None:
                self.noise = noise
            if pupil is not None:
                self.pupil = pupil
            if grid_size is not None:
                self.grid_size = grid_size
            if dx is not None:
                self.dx = dx * WFSensor_tools.ARCSEC2RAD
            if dy is not None:
                self.dy = dy * WFSensor_tools.ARCSEC2RAD

            self.field_angle = (self.dx, self.dy)
            self.sen_angle = cp.sqrt(self.field_angle[0]**2 + self.field_angle[1]**2)

            self.active_sub_aps, self.sub_aps, self.sub_aps_idx, self.sub_ap_width, self.sub_slice, self.sub_pupils = self.generate_sub_apertures(self.pupil, self.grid_size)
            

        def generate_sub_apertures(self, pupil=None, grid_size=None):
            if grid_size is None:
                grid_size = params.get("grid_size")
            if pupil is None:
                pupil = Pupil_tools.generate_pupil(grid_size=grid_size)

            sub_aps = aotools.wfs.findActiveSubaps(self.n_sub, pupil, 0.6)
            sub_aps = cp.asarray(sub_aps)
            active_sub_aps = sub_aps.shape[0]

            sub_aps_idx = (sub_aps/(grid_size/self.n_sub)).astype(int)

            sub_ap_width = grid_size/self.n_sub

            sub_slice = [(slice(i[0], i[0]+int(sub_ap_width)+1), slice(i[1], i[1]+int(sub_ap_width)+1)) for i in (sub_aps_idx.get()*sub_ap_width).astype(int)]

            sub_pupils = cp.asarray([pupil[i] for i in sub_slice])

            return active_sub_aps, sub_aps, sub_aps_idx, sub_ap_width, sub_slice, sub_pupils

        # for each subaperture find the centroid and generate an image per input phase map in phase map list 
        def measure(self, pupil=None, phase_map=None, phase_map_alt=None, poke_amplitude=None, pad=None):
            if poke_amplitude is None:
                poke_amplitude = params.get("poke_amplitude")
            if pad is None:
                pad = params.get("field_padding")
            if pupil is None:
                pupil = Pupil_tools.generate_pupil()

            n_map = phase_map.shape[0]
            n_subaps = self.sub_aps_idx.shape[0]

            h = int(self.sub_ap_width) + 1
            w = int(self.sub_ap_width) + 1
            yy = cp.arange(h)[:, None]
            xx = cp.arange(w)[None, :]

            top_left = ((self.sub_aps_idx * self.sub_ap_width).astype(int))
            top_left = cp.asarray(top_left)

            ys = top_left[:, 0, None, None] + yy[None, :, :]
            xs = top_left[:, 1, None, None] + xx[None, :, :]

            # phase maps: shape (n_map, N, N)
            phase_maps = (4.0 * cp.pi / self.wavelength) * phase_map * pupil[None, :, :]

            # extract subaperture phases: shape (n_map, n_sub, h, w)
            sub_imgs = phase_maps[:, None, :, :][:, :, ys, xs]  # broadcasting over phase maps

            # field -> PSF: shape (n_map, n_sub, h, w)
            sub_pupils = cp.ones_like(sub_imgs)  # if you have pupil mask per subap
            
            field = sub_pupils * cp.exp(1j * sub_imgs)
            
            pad_width = [(0, 0)] * (field.ndim - 2) + [(pad, pad), (pad, pad)]

            field_p = cp.pad(field, pad_width, mode='constant')

            # Now do FFT on the last two axes (works for any number of leading axes)
            F = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(field_p, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
            I = cp.abs(F) ** 2
            # normalize per subap per map
            denom = cp.sum(I, axis=(-2, -1), keepdims=True)
            # avoid division by zero
            denom = cp.where(denom == 0, 1.0, denom)
            I = I / denom  # shape (n_map, n_sub, h_p, w_p)

            # centroids (y,x) in pixels
            h_p, w_p = I.shape[-2], I.shape[-1]
            x_coords = cp.arange(w_p, dtype=cp.float32)[None, None, None, :]
            y_coords = cp.arange(h_p, dtype=cp.float32)[None, None, :, None]
            cx = cp.sum(I * x_coords, axis=(-2, -1))
            cy = cp.sum(I * y_coords, axis=(-2, -1))
            centroids = cp.stack((cy, cx), axis=-1)  # (n_map, n_sub, 2)

            # displacement from subap center (in pixels)
            center = cp.array([(h_p - 1) / 2.0, (w_p - 1) / 2.0], dtype=centroids.dtype)
            slopes = centroids - center[None, None, :]

            # stitch sensor images: compute H,W
            max_y = int(cp.max(self.sub_aps_idx[:, 0])) + 1
            max_x = int(cp.max(self.sub_aps_idx[:, 1])) + 1
            H = max_y * h_p
            W = max_x * w_p
            sensor_image = cp.zeros((n_map, H, W), dtype=I.dtype)

            # build y/x grids robustly
            iy = self.sub_aps_idx[:, 0].astype(cp.int32)   # (n_sub,)
            ix = self.sub_aps_idx[:, 1].astype(cp.int32)   # (n_sub,)

            # base offsets (1, n_sub, 1, 1)
            yy_base = (iy[None, :, None, None] * h_p).astype(cp.int32)
            xx_base = (ix[None, :, None, None] * w_p).astype(cp.int32)

            # intra-subap pixel indices (1, 1, h_p, 1) and (1,1,1,w_p)
            yy_pix = cp.arange(h_p, dtype=cp.int32)[None, None, :, None]
            xx_pix = cp.arange(w_p, dtype=cp.int32)[None, None, None, :]

            # sum to get (1, n_sub, h_p, w_p)
            yy_grid = yy_base + yy_pix        # (1, n_sub, h_p, 1) -> will broadcast when adding xx
            xx_grid = xx_base + xx_pix        # (1, n_sub, 1, w_p)

            # broadcast to (n_map, n_sub, h_p, w_p)
            yy_grid = cp.broadcast_to(yy_grid, (n_map, n_subaps, h_p, w_p))
            xx_grid = cp.broadcast_to(xx_grid, (n_map, n_subaps, h_p, w_p))

            # place intensities into sensor images using advanced indexing
            sensor_image[cp.arange(n_map)[:, None, None, None], yy_grid, xx_grid] = I

            return centroids, slopes, sensor_image

    class PyramidWFS:
        def __init__(self, n_pixels, wavelength, modulation=1.0):
            self.n_pixels = n_pixels
            self.wavelength = wavelength
            self.modulation = modulation
            self.n_slopes = 2 * n_pixels**2

        def measure(self, phase):
            # TODO
            #slopes = pyramid_response(phase, self.n_pixels, self.modulation)
            return #slopes



if __name__ == "__main__":
    import argparse
    import inspect
    import sys

    classes = {
        name: cls
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
        if not name.startswith("_")
    }

    # Collect all methods of all classes
    functions = {}
    for cls_name, cls in classes.items():
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if func.__module__ == __name__ and not name.startswith("_"):
                # store with "ClassName.method" as key
                functions[f"{cls_name}.{name}"] = func

    # Create parser
    parser = argparse.ArgumentParser(description="Run any function from this file")
    parser.add_argument("--list-functions", action="store_true", help="List available functions")
    parser.add_argument("function", nargs="?", choices=functions.keys(), help="Function to run")

    # AO parameters with defaults from config
    parser.add_argument("--telescope_diameter", type=float, default=config.get("telescope_diameter"))
    parser.add_argument("--telescope_center_obscuration", type=float, default=config.get("telescope_center_obscuration"))
    parser.add_argument("--wfs_lambda", type=float, default=config.get("wfs_lambda"))
    parser.add_argument("--science_lambda", type=float, default=config.get("science_lambda"))
    parser.add_argument("--r0", type=float, default=config.get("r0"))
    parser.add_argument("--L0", type=float, default=config.get("L0"))
    parser.add_argument("--Vwind", type=float, default=config.get("Vwind"))
    parser.add_argument("--actuators", type=int, default=config.get("actuators"))
    parser.add_argument("--sub_apertures", type=int, default=config.get("sub_apertures"))
    parser.add_argument("--frame_rate", type=int, default=config.get("frame_rate"))
    parser.add_argument("--grid_size", type=int, default=config.get("grid_size"))
    parser.add_argument("--field_padding", type=int, default=config.get("field_padding"))
    parser.add_argument("--poke_amplitude", type=float, default=config.get("poke_amplitude"))
    parser.add_argument("--random_seed", type=int, default=config.get("random_seed"))
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.set_defaults(use_gpu=config.get("use_gpu", False))
    parser.add_argument("--data_path", type=str, default=config.get("data_path"))
    
    # Extra key=value arguments
    parser.add_argument("--args", nargs="*", help="Additional key=value arguments for the function")

    args, unknown = parser.parse_known_args()

    if args.list_functions:
        print("Available functions:")
        for name in functions.keys():
            print("-", name)
        sys.exit()

    # Update params from CLI
    for key, value in vars(args).items():
        if value is not None and key != "function" and key != "args":
            params[key] = value

    # Parse extra key=value args
    kwargs = {}
    if args.args:
        for item in args.args:
            key, value = item.split("=")
            try:
                value = eval(value)
            except:
                pass
            kwargs[key] = value

    # 5 Call the chosen function
    result = functions[args.function](**kwargs)
    print("Result:", result)
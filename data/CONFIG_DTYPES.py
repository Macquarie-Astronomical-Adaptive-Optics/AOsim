CONFIG_DTYPES = {
    "telescope_diameter": float,
    "telescope_center_obscuration": float,
    "wfs_lambda": float,
    "science_lambda": float,
    "r0_total": float,
    "N": float,
    "wind": float,
    "actuators": int,
    "sub_apertures": int,
    "frame_rate": int,
    "grid_size": int,
    "field_padding": int,
    "poke_amplitude": float,
    "random_seed": int,
    "use_gpu": bool,
    "data_path": str,
    "altitude": float,
    "name": str,
    "altitude_m": float,
    "weight": float,
    "L0": float,
    "wind": list,         # expects JSON like [6.0, 1.0]
    "seed_offset": int,
}

def enforce_config_types(config):
    for key, dtype in CONFIG_DTYPES.items():
        if key in config:
            try:
                if dtype is bool:
                    config[key] = bool(config[key])
                else:
                    config[key] = dtype(config[key])
            except Exception:
                print(f"Warning: failed to convert {key}={config[key]} to {dtype.__name__}")
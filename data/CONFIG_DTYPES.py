CONFIG_DTYPES = {
    "telescope_diameter": float,
    "telescope_center_obscuration": float,
    "wfs_lambda": float,
    "science_lambda": float,
    "r0": float,
    "L0": float,
    "Vwind": float,
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
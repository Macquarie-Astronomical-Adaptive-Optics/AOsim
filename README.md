# AOsim

A PySide6 + CuPy adaptive optics simulation GUI.

## Quick start

```bash
# (Recommended) create a venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (you may need the CUDA-specific CuPy build)
pip install -r requirements.txt

# Run
python main.py --log-console
```

## Config

- Default config: `config_default.json`
- Override any key from the command line, e.g.:

```bash
python main.py --grid_size 512 --actuators 35 --no-gpu
```

If you pass `--config path/to/file.json`, that file is loaded instead of the default.

## Project layout

- `main.py`: GUI entrypoint
- `scripts/`: simulation + GUI tabs/widgets
- `turbulence/`: example turbulence profiles
- `data/`: shared dtype/config helpers

## Notes

- The simulation is GPU-first (CuPy). CPU-only mode is best-effort and mainly intended
  for development/UI testing.
- Enable the Qt log console (`--log-console`) to capture `print()` output and exceptions.

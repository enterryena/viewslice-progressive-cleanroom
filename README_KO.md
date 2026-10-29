# ViewSlice M0.2.1 - SciPy hotfix (Windows)

Error: `ModuleNotFoundError: No module named scipy`

Root cause: extension `setup.py` installs numpy, trimesh, pillow and psutil, but omits SciPy. `requirements.txt` lists SciPy for weight-transfer features.

## Recover current installation
1. Close Modly.
2. Double-click `FIX_SCIPY_WINDOWS.bat`.
3. If the extension was installed in a custom path, the script asks for the extension directory (the one with `manifest.json` and `venv\Scripts\python.exe`).
4. On SUCCESS, restart Modly and retry the ViewSlice node.

The script installs ONLY into ViewSlice's isolated `venv` using that exact `python.exe`, NOT system Python, NOT TRELLIS's environment. Internet is needed for pip.

## Prevent recurrence in new installs
Replace the GitHub repository's root `setup.py` with this hotfix `setup.py`, then commit the change. Modly's auto setup may not rerun on an already installed extension: for that installation, use the BAT above.

No source 3D models, texture assets or other extensions are changed.

Note: Windows Modly user-data extensions default to an `extensions` folder, but Modly settings can override the location.

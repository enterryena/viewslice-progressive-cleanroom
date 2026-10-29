@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
TITLE ViewSlice M0.2.1 - SciPy repair
ECHO.
ECHO ViewSlice repair: install SciPy in the extension-specific venv only.
ECHO No Modly/TRELLIS system environment will be changed.
ECHO.
set "PY="
for %%D in ("%APPDATA%\modly\extensions\viewslice-progressive-cleanroom" "%APPDATA%\Modly\extensions\viewslice-progressive-cleanroom" "%LOCALAPPDATA%\modly\extensions\viewslice-progressive-cleanroom" "%LOCALAPPDATA%\Modly\extensions\viewslice-progressive-cleanroom") do (
  if exist "%%~D\venv\Scripts\python.exe" set "PY=%%~D\venv\Scripts\python.exe"
)
if defined PY goto :install
ECHO Cannot find the extension automatically. Modly may be using a custom extension directory.
ECHO Open the ViewSlice extension folder and copy its path from the Explorer address bar.
ECHO That folder must contain manifest.json and venv\Scripts\python.exe.
ECHO.
set /p "EXT=Paste ViewSlice extension folder path (or press Enter to cancel): "
if not defined EXT goto :notfound
if not exist "%EXT%\manifest.json" goto :notfound
if not exist "%EXT%\venv\Scripts\python.exe" goto :notfound
set "PY=%EXT%\venv\Scripts\python.exe"
:install
ECHO.
ECHO Target interpreter: "%PY%"
ECHO Installing scipy into this environment...
"%PY%" -m pip install "scipy>=1.11,<2"
if errorlevel 1 goto :failed
"%PY%" -c "import scipy; print('SciPy installed:', scipy.__version__)"
if errorlevel 1 goto :failed
ECHO.
ECHO SUCCESS. Close and reopen Modly, then retry the ViewSlice node.
pause
exit /b 0
:notfound
ECHO Extension not found. No files changed. Please send a screenshot of Modly Settings (extension directory).
pause
exit /b 2
:failed
ECHO Installation failed. Please copy the lines above and share them.
pause
exit /b 1

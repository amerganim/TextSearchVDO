@echo off
rem Launcher that can talk back.
rem
rem TextSearchVDO.vbs is the one to double-click: it opens no console at all.
rem This one exists because cmd.exe can print, so when something is wrong
rem before the window appears there is somewhere for the error to go. The
rem console flash you see with this file is cmd.exe itself, not the app, and
rem no amount of pythonw can suppress it from inside a batch file.

cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo.
    echo   TextSearchVDO is not set up yet.
    echo.
    echo   Run this once, from this folder:
    echo.
    echo     py -3.14 -m venv .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    echo     .venv\Scripts\python -m tsv setup
    echo.
    pause
    exit /b 1
)

set "HAVE_DETECTOR="
if exist "data\models\yolo11n.onnx"  set "HAVE_DETECTOR=1"
if exist "data\models\yolox_tiny.onnx" set "HAVE_DETECTOR=1"
if exist "data\models\yolox_s.onnx"  set "HAVE_DETECTOR=1"

if not defined HAVE_DETECTOR (
    echo.
    echo   No detection model found in data\models.
    echo.
    echo   The app will still find movement and let you scrub the timeline,
    echo   but it cannot recognise objects or search by description.
    echo.
    echo   To add one:  .venv\Scripts\python -m tsv setup
    echo.
    timeout /t 6 >nul
)

start "" ".venv\Scripts\pythonw.exe" -m tsv app

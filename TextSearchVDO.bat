@echo off
rem Double-click launcher for the desktop window.
rem
rem Uses pythonw.exe rather than python.exe so no console window appears
rem behind the app. If something goes wrong before the window opens there is
rem nowhere for the error to go, so the setup checks happen here, in a shell
rem that can still print.

cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo.
    echo   TextSearchVDO is not set up yet.
    echo.
    echo   Run this once, from this folder:
    echo.
    echo     py -3.14 -m venv .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "data\models\yolo11n.onnx" (
    echo.
    echo   No detection model found in data\models.
    echo.
    echo   The app will still find movement and let you scrub the timeline,
    echo   but it cannot recognise objects or search by description.
    echo.
    echo   To add them, see "Getting a detector" in README.md
    echo.
    timeout /t 6 >nul
)

start "" ".venv\Scripts\pythonw.exe" -m tsv app

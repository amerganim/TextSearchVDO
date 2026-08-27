@echo off
rem First-run setup. Double-click this once, then use TextSearchVDO.bat.
rem
rem Deliberately a console window rather than a silent one: this downloads
rem well over a gigabyte and can take a while, and a progress log is the only
rem honest way to show that.

cd /d "%~dp0"
title TextSearchVDO setup

echo.
echo   TextSearchVDO setup
echo   -------------------
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo   Python was not found.
    echo.
    echo   Install Python 3.12 or newer from python.org, tick "Add to PATH",
    echo   then run this again.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo   Creating the application environment...
    py -m venv .venv
    if errorlevel 1 (
        echo   Could not create .venv. See the error above.
        pause
        exit /b 1
    )
)

echo   Installing what the app needs to run...
.venv\Scripts\python -m pip install --upgrade --quiet pip
.venv\Scripts\python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo.
    echo   Installing the requirements failed. See the error above.
    pause
    exit /b 1
)

echo.
echo   Fetching the models. This is the long part - over a gigabyte, and it
echo   builds a temporary toolchain it throws away afterwards.
echo.

.venv\Scripts\python -m tsv setup
if errorlevel 1 (
    echo.
    echo   Some parts did not install. The app still runs with less of it.
    pause
    exit /b 1
)

echo.
echo   Done. Start the app with TextSearchVDO.bat
echo.
pause

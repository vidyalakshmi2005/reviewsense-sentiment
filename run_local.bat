@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project environment...
    py -m venv .venv
    if errorlevel 1 goto setup_error
)

".venv\Scripts\python.exe" -c "import tensorflow" >nul 2>&1
if errorlevel 1 (
    echo Installing project libraries. This can take several minutes...
    ".venv\Scripts\python.exe" -m pip install -r requirements-desktop.txt
    if errorlevel 1 goto setup_error
)

".venv\Scripts\python.exe" desktop_app.py
exit /b %errorlevel%

:setup_error
echo Setup did not finish. Check the message above, fix the issue, then run this file again.
pause
exit /b 1

@echo off
rem Launch the site-neighbors GUI tool. Output files go to the repo root by default.
cd /d "%~dp0.."
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw "%~dp0neighbors_app.py"
) else (
    python "%~dp0neighbors_app.py"
    if errorlevel 1 pause
)

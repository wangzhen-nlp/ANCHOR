@echo off
rem Launch the NE-to-site info query GUI.
cd /d "%~dp0.."
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw "%~dp0ne_site_app.py"
) else (
    python "%~dp0ne_site_app.py"
    if errorlevel 1 pause
)

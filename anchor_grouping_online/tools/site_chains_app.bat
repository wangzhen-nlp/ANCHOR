@echo off
rem Launch the site_chains upstream/downstream/parallel site query GUI.
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw "%~dp0site_chains_app.py"
) else (
    python "%~dp0site_chains_app.py"
    if errorlevel 1 pause
)

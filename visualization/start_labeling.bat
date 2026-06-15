@echo off
REM Windows one-click launcher. The real logic lives in start_labeling.ps1
REM to avoid cmd.exe escaping issues with inline PowerShell.
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_labeling.ps1"
set "exitCode=%ERRORLEVEL%"

if not "%exitCode%"=="0" (
    echo.
    echo 启动失败，请查看上方 PowerShell 错误信息。
    pause
    exit /b %exitCode%
)

endlocal

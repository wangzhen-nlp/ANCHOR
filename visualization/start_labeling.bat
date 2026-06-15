@echo off
REM Windows one-click launcher. The real logic lives in resources\start_labeling.ps1
REM to avoid cmd.exe escaping issues with inline PowerShell.
REM 切到 UTF-8 控制台，避免下面中文提示在 GBK/英文码页下乱码（本文件须存为 UTF-8 无 BOM）。
chcp 65001 >nul
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0resources\start_labeling.ps1"
set "exitCode=%ERRORLEVEL%"

echo.
if "%exitCode%"=="0" (
    echo 服务已退出。
) else (
    echo 服务已停止（退出码 %exitCode%）。手动 Ctrl-C / 关窗口属正常；若是报错，请看上方 PowerShell 信息。
)
pause

endlocal

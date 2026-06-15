@echo off
REM Windows 一键启动标注（双击本文件）。无需 Python、无需服务器，用系统自带 PowerShell。
REM 它做三件事：① 把 data\*.jsonl 刷成 data.js；② 把 ne_graph.json 刷成 ne_graph.js（若有）；
REM ③ 用默认浏览器打开总览页（file://）。data\ 里的 jsonl 有增改后，重新双击即可。
cd /d "%~dp0"

REM (1) data\*.jsonl -> data.js（自动加载，省去手动选文件）
if exist "data\" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "Set-Location -LiteralPath '%~dp0';" ^
      "$files = Get-ChildItem -Path 'data' -Filter '*.jsonl' -File;" ^
      "if ($files) {" ^
      "  $lines = foreach ($f in $files) { [IO.File]::ReadAllLines($f.FullName,[Text.Encoding]::UTF8) ^| Where-Object { $_.Trim() -ne '' } };" ^
      "  [IO.File]::WriteAllText((Join-Path (Get-Location) 'data.js'), 'window.FAULT_GROUPS_DATA=[' + ($lines -join ',') + '];', (New-Object Text.UTF8Encoding($false)));" ^
      "  Write-Host ('已从 ' + $files.Count + ' 个 jsonl 生成 data.js')" ^
      "} else { Write-Host '提示：data 下没有 .jsonl，将不自动加载故障组（可在页面手动选择）。' }"
) else (
    echo 提示：未发现 data 目录，将不自动加载故障组（可在页面手动选择）。
)

REM (2) ne_graph.json -> ne_graph.js（可选）
if exist "ne_graph.json" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "Set-Location -LiteralPath '%~dp0';" ^
      "$j = [IO.File]::ReadAllText((Join-Path (Get-Location) 'ne_graph.json'),[Text.Encoding]::UTF8);" ^
      "[IO.File]::WriteAllText((Join-Path (Get-Location) 'ne_graph.js'),'window.NE_GRAPH_DATA=' + $j + ';',(New-Object Text.UTF8Encoding($false)));" ^
      "Write-Host '已从 ne_graph.json 生成 ne_graph.js'"
)

REM (3) 打开总览页（file://，默认浏览器）
start "" "ne_propagation_labeling_browser.html"

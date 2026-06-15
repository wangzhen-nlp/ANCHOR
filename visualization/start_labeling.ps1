# 微波群障根因标注 · 启动逻辑（由 start_labeling.bat 调用）。无需 Python、无需服务器。
# ① data\*.jsonl -> data.js  ② ne_graph.json -> ne_graph.js（若有）  ③ 打开总览页（file://）
$ErrorActionPreference = 'Stop'
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $dir

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# ① data\*.jsonl -> data.js（自动加载，省去手动选文件）
$dataDir = Join-Path $dir 'data'
$dataJs = Join-Path $dir 'data.js'
if (Test-Path -LiteralPath $dataDir) {
    $files = Get-ChildItem -LiteralPath $dataDir -Filter '*.jsonl' -File
    if ($files) {
        $lines = foreach ($f in $files) {
            [System.IO.File]::ReadAllLines($f.FullName, [System.Text.Encoding]::UTF8) |
                Where-Object { $_.Trim() -ne '' }
        }
        $body = 'window.FAULT_GROUPS_DATA=[' + ($lines -join ',') + '];'
        [System.IO.File]::WriteAllText($dataJs, $body, $utf8NoBom)
        Write-Host ("已从 {0} 个 jsonl 生成 data.js" -f $files.Count)
    } else {
        if (Test-Path -LiteralPath $dataJs) {
            Remove-Item -LiteralPath $dataJs -Force
        }
        Write-Host '提示：data 下没有 .jsonl，将不自动加载故障组（可在页面手动选择）。'
    }
} else {
    if (Test-Path -LiteralPath $dataJs) {
        Remove-Item -LiteralPath $dataJs -Force
    }
    Write-Host '提示：未发现 data 目录，将不自动加载故障组（可在页面手动选择）。'
}

# ② ne_graph.json -> ne_graph.js（可选）
$ng = Join-Path $dir 'ne_graph.json'
$ngJs = Join-Path $dir 'ne_graph.js'
if (Test-Path -LiteralPath $ng) {
    $json = [System.IO.File]::ReadAllText($ng, [System.Text.Encoding]::UTF8)
    [System.IO.File]::WriteAllText($ngJs, 'window.NE_GRAPH_DATA=' + $json + ';', $utf8NoBom)
    Write-Host '已从 ne_graph.json 生成 ne_graph.js'
} elseif (Test-Path -LiteralPath $ngJs) {
    Remove-Item -LiteralPath $ngJs -Force
    Write-Host '提示：未发现 ne_graph.json，已移除旧的 ne_graph.js。'
}

# ③ 打开总览页（默认浏览器）
Start-Process (Join-Path $dir 'ne_propagation_labeling_browser.html')

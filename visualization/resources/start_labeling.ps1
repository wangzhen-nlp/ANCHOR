# 微波群障根因标注 · 本地服务（由 start_labeling.bat 调用）。无需 Python、无需管理员。
# 用 TcpListener 起一个极简 HTTP 服务（仅 127.0.0.1，回环不需要 urlacl/管理员），提供：
#   GET  /list              -> data\ 下所有 *.jsonl 文件名（JSON 数组）
#   POST /save?file=<名>    -> 用请求体覆盖 data\<名>（实时回写单个故障组文件）
#   其它 GET                -> 当前目录静态文件（html / jsonl / json 等，UTF-8 文本）
# 启动后自动用默认浏览器打开总览页。关闭：在本窗口按 Ctrl-C 或直接关窗口。
$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path   # 本脚本所在目录（resources/）
$root = Split-Path -Parent $scriptDir                          # 顶层目录（start_labeling.bat 所在）
$resName = Split-Path -Leaf $scriptDir                         # 资源子目录名（默认 resources）
$dataDir = Join-Path $root 'data'                              # 故障组数据在顶层 data/
# 服务根设为顶层 $root，这样既能服务 resources\ 下的页面，也能服务 data\ 下的 jsonl。

# 旧的 file:// 注入文件若残留会干扰服务模式（页面会优先用它们），删掉以走实时文件（它们和页面同在 resources\）
foreach ($leftover in @('data.js', 'ne_graph.js')) {
    $p = Join-Path $scriptDir $leftover
    if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force }
}

function Get-FreePort {
    param([int]$Start = 8770, [int]$Tries = 50)
    for ($i = 0; $i -lt $Tries; $i++) {
        $p = $Start + $i
        try {
            $t = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $p)
            $t.Start(); $t.Stop(); return $p
        } catch { }
    }
    throw "在 $Start~$($Start + $Tries - 1) 范围内找不到可用端口"
}

function Get-ContentType {
    param([string]$Path)
    switch ([System.IO.Path]::GetExtension($Path).ToLowerInvariant()) {
        '.html' { 'text/html; charset=utf-8' }
        '.htm'  { 'text/html; charset=utf-8' }
        '.js'   { 'application/javascript; charset=utf-8' }
        '.json' { 'application/json; charset=utf-8' }
        '.jsonl' { 'application/json; charset=utf-8' }
        '.css'  { 'text/css; charset=utf-8' }
        default { 'application/octet-stream' }
    }
}

function Send-Response {
    param($Stream, [int]$Status, [string]$Reason, [byte[]]$Body, [string]$ContentType = 'text/plain; charset=utf-8')
    if ($null -eq $Body) { $Body = [byte[]]@() }
    $head = "HTTP/1.1 $Status $Reason`r`n" +
            "Content-Type: $ContentType`r`n" +
            "Content-Length: $($Body.Length)`r`n" +
            "Cache-Control: no-store`r`n" +
            "Connection: close`r`n`r`n"
    $headBytes = [System.Text.Encoding]::ASCII.GetBytes($head)
    $Stream.Write($headBytes, 0, $headBytes.Length)
    if ($Body.Length -gt 0) { $Stream.Write($Body, 0, $Body.Length) }
    $Stream.Flush()
}

function Read-Request {
    param($Stream)
    # 读到 CRLFCRLF 为止当作请求头；再按 Content-Length 读 body
    $bytes = New-Object System.Collections.Generic.List[byte]
    $last = @(0, 0, 0, 0)
    while ($true) {
        $b = $Stream.ReadByte()
        if ($b -lt 0) { break }
        $bytes.Add([byte]$b)
        $last = @($last[1], $last[2], $last[3], $b)
        if ($last[0] -eq 13 -and $last[1] -eq 10 -and $last[2] -eq 13 -and $last[3] -eq 10) { break }
    }
    $headerText = [System.Text.Encoding]::ASCII.GetString($bytes.ToArray())
    $lines = $headerText -split "`r`n"
    $requestLine = if ($lines.Length -gt 0) { $lines[0] } else { '' }
    $parts = $requestLine -split ' '
    $method = if ($parts.Length -gt 0) { $parts[0] } else { '' }
    $target = if ($parts.Length -gt 1) { $parts[1] } else { '/' }
    $contentLength = 0
    foreach ($line in $lines) {
        if ($line -match '^(?i)Content-Length:\s*(\d+)') { $contentLength = [int]$Matches[1] }
    }
    $body = ''
    if ($contentLength -gt 0) {
        $buf = New-Object byte[] $contentLength
        $read = 0
        while ($read -lt $contentLength) {
            $n = $Stream.Read($buf, $read, $contentLength - $read)
            if ($n -le 0) { break }
            $read += $n
        }
        $body = [System.Text.Encoding]::UTF8.GetString($buf, 0, $read)
    }
    return [pscustomobject]@{ Method = $method; Target = $target; Body = $body }
}

# 只允许纯文件名，挡住路径穿越
function Get-SafeName {
    param([string]$Name)
    if ([string]::IsNullOrWhiteSpace($Name)) { return $null }
    $decoded = [System.Uri]::UnescapeDataString($Name)
    if ($decoded -ne [System.IO.Path]::GetFileName($decoded)) { return $null }
    if ($decoded -match '[\\/]' -or $decoded.Contains('..')) { return $null }
    return $decoded
}

function Get-QueryParam {
    param([string]$Query, [string]$Name)
    foreach ($kv in ($Query -split '&')) {
        $pair = $kv -split '=', 2
        if ($pair.Length -eq 2 -and $pair[0] -eq $Name) { return $pair[1] }
    }
    return $null
}

$port = Get-FreePort
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
$listener.Start()

$entryUrl = "http://127.0.0.1:$port/$resName/ne_propagation_labeling_browser.html"
Write-Host ("=" * 56)
Write-Host "  微波群障根因标注 · 本地服务已启动"
Write-Host "  地址：$entryUrl"
Write-Host "  目录：$root"
Write-Host "  实时回写：标注一变即写回 data\ 对应的 jsonl"
Write-Host "  关闭：关掉所有标注页后服务会自动退出；也可在本窗口按 Ctrl-C 或直接关窗口"
Write-Host ("=" * 56)
Start-Process $entryUrl

# 心跳联动：每个页面带 pageId 请求 /ping；关闭时尽量 /leave；失联页面由超时清理兜底。
# everConnected 门闩：首个请求到达前不计空闲超时，避免冷启动浏览器较慢时服务被提前关掉。
$pageTtlSec = 90
$shutdownGraceSec = 10
$activePages = @{}
$noActiveSince = $null
$everConnected = $false

try {
    while ($true) {
        $now = Get-Date
        foreach ($pageId in @($activePages.Keys)) {
            if (($now - $activePages[$pageId]).TotalSeconds -gt $pageTtlSec) {
                $activePages.Remove($pageId) | Out-Null
            }
        }
        if ($activePages.Count -gt 0) {
            $noActiveSince = $null
        } elseif ($everConnected -and $null -eq $noActiveSince) {
            $noActiveSince = $now
        }

        # Poll 在有连接到达时立即返回（零延迟），否则最多等 0.5s 再查空闲——
        # 不能用固定 Start-Sleep，否则每个请求都被拖延，逐个加载 data 文件时会很慢。
        if (-not $listener.Server.Poll(500000, [System.Net.Sockets.SelectMode]::SelectRead)) {
            if ($everConnected -and $activePages.Count -eq 0 -and $null -ne $noActiveSince -and (($now - $noActiveSince).TotalSeconds -gt $shutdownGraceSec)) {
                Write-Host "所有标注页已关闭，本地服务自动退出。"
                break
            }
            continue
        }
        $client = $listener.AcceptTcpClient()
        $client.ReceiveTimeout = 5000   # 防止空连接(预连接)在读请求时永久阻塞单线程服务
        $everConnected = $true
        $stream = $client.GetStream()
        try {
            $req = Read-Request $stream
            $path = ($req.Target -split '\?')[0]
            $query = if ($req.Target.Contains('?')) { ($req.Target -split '\?', 2)[1] } else { '' }

            if ($req.Method -eq 'GET' -and $path -eq '/ping') {
                $pageId = Get-QueryParam $query 'pageId'
                if (-not [string]::IsNullOrWhiteSpace($pageId)) {
                    $activePages[$pageId] = Get-Date
                    $noActiveSince = $null
                }
                $json = '{"ok":true,"activePages":' + $activePages.Count + '}'
                Send-Response $stream 200 'OK' ([System.Text.Encoding]::UTF8.GetBytes($json)) 'application/json; charset=utf-8'
            }
            elseif (($req.Method -eq 'GET' -or $req.Method -eq 'POST') -and $path -eq '/leave') {
                $pageId = Get-QueryParam $query 'pageId'
                if (-not [string]::IsNullOrWhiteSpace($pageId)) {
                    $activePages.Remove($pageId) | Out-Null
                }
                if ($activePages.Count -eq 0) { $noActiveSince = Get-Date }
                $json = '{"ok":true,"activePages":' + $activePages.Count + '}'
                Send-Response $stream 200 'OK' ([System.Text.Encoding]::UTF8.GetBytes($json)) 'application/json; charset=utf-8'
            }
            elseif ($req.Method -eq 'GET' -and $path -eq '/list') {
                $names = @()
                if (Test-Path -LiteralPath $dataDir) {
                    $names = @(Get-ChildItem -LiteralPath $dataDir -Filter '*.jsonl' -File | ForEach-Object { $_.Name })
                }
                # 手工拼 JSON 数组，避开 ConvertTo-Json 把单元素数组拆成裸字符串的坑
                $escaped = $names | ForEach-Object { '"' + ($_ -replace '\\', '\\' -replace '"', '\"') + '"' }
                $json = '[' + ($escaped -join ',') + ']'
                Send-Response $stream 200 'OK' ([System.Text.Encoding]::UTF8.GetBytes($json)) 'application/json; charset=utf-8'
            }
            elseif ($req.Method -eq 'POST' -and $path -eq '/save') {
                $fileParam = Get-QueryParam $query 'file'
                $name = Get-SafeName $fileParam
                if (-not $name) {
                    Send-Response $stream 400 'Bad Request' ([System.Text.Encoding]::UTF8.GetBytes('invalid file name'))
                } else {
                    if (-not (Test-Path -LiteralPath $dataDir)) { New-Item -ItemType Directory -Path $dataDir | Out-Null }
                    $target = Join-Path $dataDir $name
                    $content = $req.Body
                    if (-not $content.EndsWith("`n")) { $content += "`n" }
                    [System.IO.File]::WriteAllText($target, $content, (New-Object System.Text.UTF8Encoding($false)))
                    Send-Response $stream 200 'OK' ([System.Text.Encoding]::UTF8.GetBytes('{"ok":true}')) 'application/json; charset=utf-8'
                    Write-Host ("已回写 data\{0}" -f $name)
                }
            }
            elseif ($req.Method -eq 'GET') {
                $rel = [System.Uri]::UnescapeDataString($path.TrimStart('/'))
                if ([string]::IsNullOrWhiteSpace($rel)) { $rel = "$resName/ne_propagation_labeling_browser.html" }
                $full = Join-Path $root $rel
                # 限定在服务根目录内，挡住穿越
                $fullResolved = [System.IO.Path]::GetFullPath($full)
                if (-not $fullResolved.StartsWith([System.IO.Path]::GetFullPath($root))) {
                    Send-Response $stream 403 'Forbidden' ([System.Text.Encoding]::UTF8.GetBytes('forbidden'))
                }
                elseif (Test-Path -LiteralPath $fullResolved -PathType Leaf) {
                    $fileBytes = [System.IO.File]::ReadAllBytes($fullResolved)
                    Send-Response $stream 200 'OK' $fileBytes (Get-ContentType $fullResolved)
                }
                else {
                    Send-Response $stream 404 'Not Found' ([System.Text.Encoding]::UTF8.GetBytes('not found'))
                }
            }
            else {
                Send-Response $stream 405 'Method Not Allowed' ([System.Text.Encoding]::UTF8.GetBytes('method not allowed'))
            }
        }
        catch {
            try { Send-Response $stream 500 'Internal Server Error' ([System.Text.Encoding]::UTF8.GetBytes('server error')) } catch { }
        }
        finally {
            $stream.Close(); $client.Close()
        }
    }
}
finally {
    $listener.Stop()
}

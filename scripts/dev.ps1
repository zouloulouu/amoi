param(
    [int]$ApiPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Frontend = Join-Path $Root "frontend"
$CorsOrigins = @(
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174"
) -join ","

function Test-PortListening {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $connection
}

if (Test-PortListening $ApiPort) {
    Write-Host "Port API $ApiPort deja utilise. Arrete le process existant ou change -ApiPort." -ForegroundColor Yellow
    netstat -ano | findstr ":$ApiPort"
    exit 1
}

if (Test-PortListening $FrontendPort) {
    Write-Host "Port frontend $FrontendPort deja utilise. Arrete le process existant ou change -FrontendPort." -ForegroundColor Yellow
    netstat -ano | findstr ":$FrontendPort"
    exit 1
}

$RootLiteral = $Root.Path.Replace("'", "''")
$FrontendLiteral = $Frontend.Replace("'", "''")
$CorsLiteral = $CorsOrigins.Replace("'", "''")
$ApiUrlLiteral = "http://127.0.0.1:$ApiPort"

$apiCommand = "& { Set-Location -LiteralPath '$RootLiteral'; `$env:INA_CORS_ORIGINS = '$CorsLiteral'; python -m uvicorn ina_api.main:app --host 127.0.0.1 --port $ApiPort }"
$frontCommand = "& { Set-Location -LiteralPath '$FrontendLiteral'; `$env:VITE_API_BASE_URL = '$ApiUrlLiteral'; npm.cmd run dev -- --port $FrontendPort }"

Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $apiCommand)
Start-Sleep -Seconds 2
Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $frontCommand)

Write-Host "API      http://127.0.0.1:$ApiPort" -ForegroundColor Green
Write-Host "Frontend http://127.0.0.1:$FrontendPort" -ForegroundColor Green

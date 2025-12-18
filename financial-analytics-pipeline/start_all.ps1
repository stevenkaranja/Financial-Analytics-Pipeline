# start_all.ps1 - Start All Portfolio Analytics Services

$ProjectRoot = "C:\Users\HomePC\Documents\Automation Portfolio\financial-analytics-pipeline"
$PythonExe = "C:\Users\HomePC\Documents\Automation Portfolio\venv\Scripts\python.exe"
$StreamlitExe = "C:\Users\HomePC\Documents\Automation Portfolio\venv\Scripts\streamlit.exe"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "🚀 Starting Portfolio Analytics Platform" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Kill existing processes
Write-Host "🧹 Cleaning up existing processes..." -ForegroundColor Yellow
Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process python -ErrorAction SilentlyContinue | Where-Object { 
    $_.CommandLine -like '*db_api_server*' -or 
    $_.CommandLine -like '*etl_server*' 
} | Stop-Process -Force
Start-Sleep -Seconds 2
Write-Host "✅ Cleanup complete" -ForegroundColor Green
Write-Host ""

# Check n8n Availability
Write-Host "🌐 Checking n8n Availability..." -ForegroundColor Cyan
try {
    $n8nResponse = Invoke-WebRequest -Uri "https://n8n.datastagke.com" -Method GET -TimeoutSec 5 -ErrorAction Stop -UseBasicParsing
    if ($n8nResponse.StatusCode -eq 200) {
        Write-Host "✅ n8n is accessible at https://n8n.datastagke.com" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  n8n health check failed - workflows may not work" -ForegroundColor Yellow
}
Write-Host ""

# Check Cloudflare Tunnel Status (optional - may require cloudflared CLI)
Write-Host "☁️  Checking Cloudflare Tunnel..." -ForegroundColor Cyan
$cloudflaredPath = Get-Command cloudflared -ErrorAction SilentlyContinue
if ($cloudflaredPath) {
    try {
        $tunnelStatus = & cloudflared tunnel list 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Cloudflare tunnel is configured" -ForegroundColor Green
        } else {
            Write-Host "ℹ️  Cloudflare tunnel not active (optional)" -ForegroundColor Gray
        }
    } catch {
        Write-Host "ℹ️  Cloudflare tunnel status unavailable" -ForegroundColor Gray
    }
} else {
    Write-Host "ℹ️  cloudflared not installed (Cloudflare tunnel not required)" -ForegroundColor Gray
}
Write-Host ""

# Start ETL Server
Write-Host "⚙️  Starting ETL Server (Port 5001)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
`$Host.UI.RawUI.WindowTitle = 'ETL Server (Port 5001)'
cd '$ProjectRoot\data\raw'
& '$PythonExe' etl_server.py
"@

Start-Sleep -Seconds 4

# Check ETL Server health
try {
    $etlHealth = Invoke-RestMethod -Uri "http://localhost:5001/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
    if ($etlHealth.status -eq "healthy") {
        Write-Host "✅ ETL Server is running on port $($etlHealth.port)" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ ETL Server health check failed" -ForegroundColor Red
}
Write-Host ""

# Start Flask API Server
Write-Host "🔧 Starting Flask API Server (Port 5000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
`$Host.UI.RawUI.WindowTitle = 'Flask API Server (Port 5000)'
cd '$ProjectRoot\dashboard'
& '$PythonExe' db_api_server.py
"@

Start-Sleep -Seconds 4

# Check Flask API health
try {
    $apiHealth = Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
    if ($apiHealth.status -eq "healthy") {
        Write-Host "✅ Flask API Server is running" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Flask API Server health check failed" -ForegroundColor Red
}
Write-Host ""

# Start Streamlit Dashboard
Write-Host "🎨 Starting Streamlit Dashboard (Port 8501)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
`$Host.UI.RawUI.WindowTitle = 'Streamlit Dashboard (Port 8501)'
cd '$ProjectRoot\dashboard'
& '$StreamlitExe' run app.py --server.port=8501
"@

Start-Sleep -Seconds 5

# Check Streamlit availability
try {
    $streamlitResponse = Invoke-WebRequest -Uri "http://localhost:8501" -Method GET -TimeoutSec 5 -ErrorAction Stop -UseBasicParsing
    Write-Host "✅ Streamlit Dashboard is running" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Streamlit Dashboard starting (may take a few more seconds)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "✅ Platform Started Successfully!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Local Access Points:" -ForegroundColor Yellow
Write-Host "   Dashboard:       http://localhost:8501" -ForegroundColor White
Write-Host "   Flask API:       http://localhost:5000" -ForegroundColor White
Write-Host "   ETL Server:      http://localhost:5001" -ForegroundColor White
Write-Host "   Flask Health:    http://localhost:5000/health" -ForegroundColor White
Write-Host "   ETL Health:      http://localhost:5001/health" -ForegroundColor White
Write-Host ""
Write-Host "🌐 External Services:" -ForegroundColor Yellow
Write-Host "   n8n Workflows:   https://n8n.datastagke.com" -ForegroundColor White
Write-Host ""
Write-Host "🛑 To stop all services:" -ForegroundColor Yellow
Write-Host "   Get-Process streamlit,python | Stop-Process -Force" -ForegroundColor White
Write-Host ""
Write-Host "📁 Important Paths:" -ForegroundColor Yellow
Write-Host "   Database:        $ProjectRoot\data\database\finance_data.db" -ForegroundColor White
Write-Host "   Scheduler Log:   $ProjectRoot\data\logs\scheduler.log" -ForegroundColor White
Write-Host ""
Write-Host "🔄 Data Flow:" -ForegroundColor Yellow
Write-Host "   Add Holding:     Streamlit → n8n → ETL Server → SQLite" -ForegroundColor White
Write-Host "   Delete Holding:  Streamlit → Flask API → SQLite" -ForegroundColor White
Write-Host "   Hourly Updates:  Scheduler → n8n → ETL Server → SQLite" -ForegroundColor White
Write-Host "   View Data:       Streamlit → SQLite" -ForegroundColor White
Write-Host ""

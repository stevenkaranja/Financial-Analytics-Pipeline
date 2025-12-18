# scheduler/setup_scheduler.ps1 - Setup Windows Task Scheduler for Hourly Ingestion

$ProjectRoot = "C:\Users\HomePC\Documents\Automation Portfolio\financial-analytics-pipeline"
$PythonExe = "C:\Users\HomePC\Documents\Automation Portfolio\venv\Scripts\python.exe"
$ScriptPath = "$ProjectRoot\scheduler\hourly_ingest.py"
$TaskName = "PortfolioDataIngest"

Write-Host "=" * 60
Write-Host "Setting up hourly data ingestion scheduler..." -ForegroundColor Cyan

# Check if task already exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "Task already exists. Removing old task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create task action
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $ScriptPath `
    -WorkingDirectory $ProjectRoot

# Create trigger (every hour)
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# Register task
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Hourly data ingestion for portfolio assets" `
    -User $env:USERNAME `
    -RunLevel Highest

Write-Host "Task created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Task Details:" -ForegroundColor Cyan
Write-Host "  Name: $TaskName"
Write-Host "  Frequency: Every hour"
Write-Host "  Script: $ScriptPath"
Write-Host ""
Write-Host "To manage this task:" -ForegroundColor Yellow
Write-Host "  View:   Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Start:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Stop:   Stop-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Remove: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
Write-Host ""
Write-Host "Starting initial run..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName $TaskName

Write-Host "=" * 60
Write-Host "✅ Scheduler setup complete!" -ForegroundColor Green

param(
    [Parameter(Mandatory = $true)][string]$Config,
    [string]$At = '03:00'
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
$BackupScript = Join-Path $PSScriptRoot 'scheduled_backup.py'
$ConfigPath = (Resolve-Path -LiteralPath $Config).Path
if (-not (Test-Path -LiteralPath $PythonExecutable)) { throw 'Project Python environment is missing.' }
if (-not (Get-Command restic -ErrorAction SilentlyContinue)) { throw 'Install Restic before scheduling backups.' }
# pythonw runs the task without an interactive console; exit codes remain visible in Task Scheduler.
$PythonWindowless = Join-Path $ProjectRoot '.venv/Scripts/pythonw.exe'
$TaskArguments = '"{0}" --config "{1}"' -f $BackupScript, $ConfigPath
$Action = New-ScheduledTaskAction -Execute $PythonWindowless -Argument $TaskArguments -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $At
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 24)
$Principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName 'BoneMarrowVision encrypted backup' -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description 'Verified Compose snapshot and encrypted Restic backup. Requires the user session and Docker.'

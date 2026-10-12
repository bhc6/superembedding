$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\use_d_storage.ps1"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$v5Root = 'D:\DEMO\superembedding-storage\superembedding\v5'
$trainingRun = Join-Path $v5Root 'general_stage1_v51'
$ablationRun = Join-Path $v5Root 'general_ablation'
$trainingStatus = Get-Content -LiteralPath (Join-Path $trainingRun 'status.json') -Raw | ConvertFrom-Json
$active = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -like '*superembedding*' })
if ($trainingStatus.state -ne 'completed' -and -not ($active | Where-Object { $_.CommandLine -match 'scripts[\\/]supervise_general.py|scripts[\\/]train_general.py' })) {
    if (Test-Path -LiteralPath (Join-Path $trainingRun 'STOP_AFTER_CHECKPOINT')) {
        throw 'Training pause marker exists. Remove it only when resuming the intended run.'
    }
    $started = Start-Process -FilePath $pythonPath -ArgumentList '-u','scripts/supervise_general.py','--data',"$v5Root\data_v51",'--output',$trainingRun,'--steps','2000' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput "$trainingRun\supervisor_on_d_stdout.log" -RedirectStandardError "$trainingRun\supervisor_on_d_stderr.log" -PassThru
    Write-Output "Started training supervisor PID $($started.Id) on D:"
}
$ablationStatus = Get-Content -LiteralPath (Join-Path $ablationRun 'pipeline_status.json') -Raw | ConvertFrom-Json
if ($ablationStatus.state -ne 'completed' -and -not ($active | Where-Object { $_.CommandLine -match 'scripts[\\/]run_general_ablation.py|scripts[\\/]extract_general_ablation.py|scripts[\\/]train_general_ablation.py' })) {
    if (Test-Path -LiteralPath "$v5Root\general_ablation_features\STOP_AFTER_SHARD") {
        throw 'Feature extraction pause marker exists. Remove it only when resuming.'
    }
    $started = Start-Process -FilePath $pythonPath -ArgumentList '-u','scripts/run_general_ablation.py','--data',"$v5Root\data_v51",'--root',$ablationRun -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput "$ablationRun\on_d_stdout.log" -RedirectStandardError "$ablationRun\on_d_stderr.log" -PassThru
    Write-Output "Started CPU ablation pipeline PID $($started.Id) on D:"
}

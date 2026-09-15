<#
.SYNOPSIS
    Bring up five headless clients, wait for them, and start a training run.

.DESCRIPTION
    One command for the whole sequence, because the ordering matters: a trainer
    that starts before its clients answer burns a lane's max_episode_failures
    while that client is still loading, and the lane retires for the rest of
    the run.

    Start it from a terminal, not from an agent's background task. A long run
    supervised as someone's child process dies with that process: two runs died
    at 80 and ~55 optimizer updates that way, neither having reached a
    checkpoint. Nothing was actually short of memory -- five clients plus a CPU
    trainer is 5.3 GB of 15.5 GB, with 3+ GB free measured throughout.

    --device cpu is deliberate, not a fallback. The model is 1.3M parameters
    with half of that in embedding tables, so Python dispatch and kernel launch
    overhead dominate the arithmetic; a CUDA context costs 2.1 GB against the
    638 MB this takes, and the forward pass is a few ms of a step whose wall
    clock is HTTP and the game settling. See CLAUDE.md, "Encoding stays
    batched".

.EXAMPLE
    pwsh -File scripts/start_training_run.ps1

.EXAMPLE
    pwsh -File scripts/start_training_run.ps1 -RunDir runs/experiment -TotalEpisodes 2000
#>
param(
    [string]$RunDir = 'runs/pool150-20260913',
    # Optimizer updates between checkpoints. 50 caps what an unexpected
    # death costs at about eight minutes; checkpoints are 15.6 MB each and
    # nothing prunes them, so do not set this very low for a long run.
    [int]$CheckpointEvery = 50,
    [int]$TotalEpisodes = 20000
)

$ErrorActionPreference = 'Stop'
$repo = (Get-Location).Path
$pidFile = Join-Path $repo 'logs/train/detached.pids'
# Start-Process resolves redirect paths against the caller's working
# directory rather than -WorkingDirectory, so these must be absolute.
$supervisorOut = Join-Path $repo 'logs/game_clients/supervisor.log'
$supervisorErr = Join-Path $repo 'logs/game_clients/supervisor.err.log'
$trainerOut = Join-Path $repo 'logs/train/pool150-20260913.log'
$trainerErr = Join-Path $repo 'logs/train/pool150-20260913.err.log'
New-Item -ItemType Directory -Force -Path (Join-Path $repo 'logs/train') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repo 'logs/game_clients') | Out-Null

$supervisor = Start-Process -FilePath 'pwsh' -PassThru -WindowStyle Hidden `
    -WorkingDirectory $repo `
    -ArgumentList @('-NoProfile', '-File', 'scripts/start_game_clients.ps1',
                    '-Count', '5', '-Headless') `
    -RedirectStandardOutput $supervisorOut `
    -RedirectStandardError  $supervisorErr
"supervisor pid=$($supervisor.Id)"

# Wait for every client to answer before the trainer starts, so a lane does
# not retire on max_episode_failures while its client is still loading.
$ports = 15526..15530
$deadline = (Get-Date).AddMinutes(6)
while ((Get-Date) -lt $deadline) {
    $up = 0
    foreach ($port in $ports) {
        try {
            $null = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/v1/singleplayer?format=json" `
                -TimeoutSec 3 -UseBasicParsing
            $up++
        } catch {}
    }
    if ($up -eq $ports.Count) { break }
    Start-Sleep -Seconds 5
}
if ($up -ne $ports.Count) {
    Stop-Process -Id $supervisor.Id -Force -Confirm:$false
    throw "only $up/$($ports.Count) clients came up; supervisor stopped"
}
"all $up clients healthy"

# OMP/MKL pinned to one thread so torch does not take cores back from the
# clients' frame rate, which is directly steps per second here.
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$trainerArgs = @(
    'run', 'sts2rl-train',
    '--run-dir', $RunDir,
    '--game-mode', 'custom',
    '--ports', '15526,15527,15528,15529,15530',
    '--device', 'cpu',
    '--seed-pool', 'default',
    '--holdout-seeds', 'default',
    '--checkpoint-every', "$CheckpointEvery",
    '--total-episodes', "$TotalEpisodes",
    '--allow-active-run'
)
$trainer = Start-Process -FilePath 'uv' -PassThru -WindowStyle Hidden `
    -WorkingDirectory $repo -ArgumentList $trainerArgs `
    -RedirectStandardOutput $trainerOut `
    -RedirectStandardError  $trainerErr
"trainer pid=$($trainer.Id)"

"supervisor=$($supervisor.Id)`ntrainer=$($trainer.Id)" | Set-Content -LiteralPath $pidFile -Encoding UTF8
"pids written to $pidFile"

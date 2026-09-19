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
    [int]$TotalEpisodes = 20000,
    # A behavior-cloning artifact (sts2rl-bc-train's bc_best.pt) to start the
    # encoder from instead of random weights. Only for a new run: the trainer
    # refuses it beside --resume.
    [string]$InitEncoder = ''
)

$ErrorActionPreference = 'Stop'
# Checked before any client starts: a wrong path found after six minutes of
# waiting for clients is six minutes wasted.
if ($InitEncoder -and -not (Test-Path -LiteralPath $InitEncoder)) {
    throw "init encoder not found: $InitEncoder"
}
$repo = (Get-Location).Path
$pidFile = Join-Path $repo 'logs/train/detached.pids'
# Start-Process resolves redirect paths against the caller's working
# directory rather than -WorkingDirectory, so these must be absolute.
$supervisorOut = Join-Path $repo 'logs/game_clients/supervisor.log'
$supervisorErr = Join-Path $repo 'logs/game_clients/supervisor.err.log'
# Named after the run, so two runs never share a log.
$runName = Split-Path -Leaf $RunDir
$trainerOut = Join-Path $repo "logs/train/$runName.log"
$trainerErr = Join-Path $repo "logs/train/$runName.err.log"
New-Item -ItemType Directory -Force -Path (Join-Path $repo 'logs/train') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repo 'logs/game_clients') | Out-Null

$supervisor = Start-Process -FilePath 'pwsh' -PassThru -WindowStyle Hidden `
    -WorkingDirectory $repo `
    -ArgumentList @('-NoProfile', '-File', 'scripts/start_game_clients.ps1',
                    '-Count', '5', '-Headless') `
    -RedirectStandardOutput $supervisorOut `
    -RedirectStandardError  $supervisorErr
"supervisor pid=$($supervisor.Id)"

# Wait for every client to *accept input* before the trainer starts, so a lane
# does not retire on max_episode_failures while its client is still loading.
#
# Answering a GET is not enough.  A freshly started client reports the main
# menu, with singleplayer on offer, for a minute or more before it will act on
# a click -- the request is refused with "Not on a menu screen".  Waiting on
# the GET alone let a trainer in during that window: all five lanes spent their
# reset attempts in 85 seconds and retired before a single episode.  So a
# client counts as ready once a real click goes through.
#
# The click is exactly the one reset would make first -- ``abandon_run`` when
# the menu offers it, ``singleplayer`` otherwise (``ResetController.
# _advance_menu``) -- so probing changes nothing reset would not.  Hardcoding
# ``singleplayer`` was wrong: a client holding a save shows ``continue`` and
# ``abandon_run`` instead, every probe was refused with "Option 'singleplayer'
# is not available", and the launcher gave up after eight minutes on five
# perfectly healthy clients.  A client already past the main menu (a submenu,
# a popup, or a run in progress) is ready as it stands and is not clicked.
function Get-OptionNames($state) {
    @($state.options | ForEach-Object {
        if ($_ -is [string]) { $_ } elseif ($_.enabled -ne $false) { $_.name }
    })
}
$ports = 15526..15530
$ready = @{}
$lastRefusal = @{}
$deadline = (Get-Date).AddMinutes(8)
while ((Get-Date) -lt $deadline -and $ready.Count -lt $ports.Count) {
    foreach ($port in $ports) {
        if ($ready.ContainsKey($port)) { continue }
        $uri = "http://127.0.0.1:$port/api/v1/singleplayer"
        try {
            $state = Invoke-RestMethod -Uri "$uri`?format=json" -TimeoutSec 3
            if ($state.state_type -ne 'menu' -or $state.menu_screen -ne 'main') {
                $ready[$port] = $true
                continue
            }
            $option = if ((Get-OptionNames $state) -contains 'abandon_run') {
                'abandon_run'
            } else {
                'singleplayer'
            }
            $click = Invoke-RestMethod -Uri $uri -Method Post -TimeoutSec 10 `
                -ContentType 'application/json' `
                -Body (@{ action = 'menu_select'; option = $option } | ConvertTo-Json)
            if ($click.status -eq 'ok') { $ready[$port] = $true }
            else { $lastRefusal[$port] = "$option -> $($click.error)" }
        } catch { $lastRefusal[$port] = $_.Exception.Message }
    }
    if ($ready.Count -lt $ports.Count) { Start-Sleep -Seconds 5 }
}
# Say why a client never became ready, or the next failure is as opaque as the
# last one was.
foreach ($port in $ports) {
    if (-not $ready.ContainsKey($port) -and $lastRefusal.ContainsKey($port)) {
        "client $port last refusal: $($lastRefusal[$port])"
    }
}
$up = $ready.Count
if ($up -ne $ports.Count) {
    # The whole tree: stopping the supervisor alone orphans its five clients,
    # which then hold about 3.7 GB until someone notices.
    & taskkill.exe /PID $supervisor.Id /T /F | Out-Null
    throw "only $up/$($ports.Count) clients accept input; supervisor and clients stopped"
}
"all $up clients accept input"

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
if ($InitEncoder) {
    $trainerArgs += @('--init-encoder', $InitEncoder)
}
$trainer = Start-Process -FilePath 'uv' -PassThru -WindowStyle Hidden `
    -WorkingDirectory $repo -ArgumentList $trainerArgs `
    -RedirectStandardOutput $trainerOut `
    -RedirectStandardError  $trainerErr
"trainer pid=$($trainer.Id)"

"supervisor=$($supervisor.Id)`ntrainer=$($trainer.Id)" | Set-Content -LiteralPath $pidFile -Encoding UTF8
"pids written to $pidFile"

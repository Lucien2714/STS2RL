<#
.SYNOPSIS
    Start and supervise N Slay the Spire 2 game clients for STS2RL training.

.DESCRIPTION
    Replaces start_three_game_clients.bat / start_four_game_clients.bat, which
    were the same script twice with a different instance count.

    Three things this does that the batch versions did not:

      * -Headless runs each client with Godot's headless display driver, so no
        window and no D3D12 device is created. See docs/headless.md.

      * Each client gets its own APPDATA, so each gets its own user:// save
        directory. Without this every client writes the same save files and
        they overwrite each other -- the "Cloud write failed" and "Rename
        failed" warnings in logs/game_clients/*.log are exactly that.

      * The mod's configured port is read from STS2_MCP.conf and checked
        against the port this script expects. The mod takes its port from that
        file, not from the command line, so a mismatch means the trainer talks
        to a client it did not think it was talking to.

.PARAMETER Count
    How many clients to start. Ports are assigned BasePort, BasePort+1, ...

.PARAMETER GameRoot
    Directory holding the per-client game installs.

.PARAMETER InstanceFormat
    Folder name pattern under GameRoot; {0} is the 1-based client index.

.PARAMETER BasePort
    Port expected for client 1. Each client must already have this port in its
    own mods/STS2_MCP/STS2_MCP.conf; changing a port needs a game restart.

.PARAMETER Headless
    Pass --headless to the game. Removes the window and the render device.

.PARAMETER SharedSaves
    Opt out of per-client save isolation and let every client share one APPDATA,
    which is what the batch scripts did. Only useful for reproducing the old
    behaviour.

.PARAMETER SaveRoot
    Where the per-client APPDATA directories are created.

.EXAMPLE
    pwsh -File scripts/start_game_clients.ps1 -Count 4 -Headless

.EXAMPLE
    # Reproduce the old windowed setup for an A/B throughput comparison.
    pwsh -File scripts/start_game_clients.ps1 -Count 4
#>

[CmdletBinding()]
param(
    [ValidateRange(1, 32)]
    [int]$Count = 4,

    # Start only these client indices instead of 1..Count. Useful to bring up a
    # subset while another client is busy, and to restart a single one.
    [int[]]$Indices,

    [string]$GameRoot = 'D:\sts2depot',

    [string]$InstanceFormat = 'sts2_{0}',

    [ValidateRange(1, 65535)]
    [int]$BasePort = 15526,

    [switch]$Headless,

    [switch]$SharedSaves,

    # Check installs, mod ports and save directories, print what would happen,
    # and exit without starting anything.
    [switch]$ValidateOnly,

    [string]$SaveRoot = (Join-Path $env:LOCALAPPDATA 'STS2RL\clients'),

    # Where to copy settings.save from when creating a client's save directory
    # for the first time. See Initialize-ClientSaves for why this is needed.
    [string]$SeedFrom = (Join-Path $env:APPDATA 'SlayTheSpire2'),

    [string]$LogDir = 'logs/game_clients',

    [string]$ApiHost = '127.0.0.1',

    [int]$HealthTimeoutSeconds = 3,

    [int]$CheckIntervalSeconds = 5,

    [int]$StartupGraceSeconds = 60,

    [int]$MaxFailedHealthChecks = 3,

    [int]$RestartCooldownSeconds = 5,

    # Extra arguments appended to every client's command line.
    [string[]]$ExtraGameArgs = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Log {
    param([string]$Message)
    Write-Host ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message)
}

# --------------------------------------------------------------------------
# Client description
# --------------------------------------------------------------------------

class GameClient {
    [int]$Index
    [int]$Port
    [string]$ExePath
    [string]$AppData
    [string]$StdOut
    [string]$StdErr
    [System.Diagnostics.Process]$Process
    [int]$FailedChecks
    [datetime]$StartedAt

    [string] Label() { return "client-$($this.Index)" }

    [string] HealthUrl() {
        return "http://$($script:ApiHost):$($this.Port)/api/v1/singleplayer?format=json"
    }
}

<#
    Read the port the mod will actually bind.

    The mod reads its port from STS2_MCP.conf beside its DLL and ignores the
    command line, so this is the only authoritative source. Returning $null
    means "could not determine", which the caller treats as a warning rather
    than a failure -- an older mod layout is not worth refusing to start over.
#>
function Get-ModPort {
    param([string]$InstallDir)

    $candidates = @(
        (Join-Path $InstallDir 'mods\STS2_MCP\STS2_MCP.conf'),
        (Join-Path $InstallDir 'mods\sts2mcp\STS2_MCP.conf')
    )
    foreach ($path in $candidates) {
        if (-not (Test-Path -LiteralPath $path)) { continue }
        try {
            # -Raw because the file has no trailing newline in some copies.
            $conf = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
            if ($null -ne $conf.port) { return [int]$conf.port }
        } catch {
            Write-Log "Could not parse $path : $($_.Exception.Message)"
        }
    }
    return $null
}

<#
    Give a brand-new client save directory the files it cannot do without.

    Two of them, and each missing one costs a whole training lane.

    settings.save carries mods_enabled, the mod list and seen_ea_disclaimer. A
    client started against an empty APPDATA loads no mods at all: the game logs
    "Skipping loading mod STS2_MCP, user has not yet seen the mods warning" and
    then serves no API, so the trainer sees a client that starts and never
    answers.

    progress.save carries the account's unlocks, including which epochs have
    been revealed. Without it the main menu offers only settings and quit --
    timeline is blocked with "manual_epoch_reveal_required" and there is no
    singleplayer entry at all -- so reset fails with "Menu option
    'singleplayer' is not enabled" and the lane retires after
    max_episode_failures. That is the expensive shape of this one: the API
    answers, this script's health check passes, and only the trainer's metrics
    say anything is wrong.

    Steam Cloud is why neither can be assumed to arrive on its own. It syncs
    the profile and the run history down into a new save directory, but never
    settings.save, and it raced four clients coming up at once on
    progress.save: two of them read the 5 KB default before the 200 KB cloud
    copy landed.

    Run state -- current_run.save and the history -- is deliberately not
    copied. The point of separate save directories is that clients diverge, and
    seeding a live run would start them as copies of one another. Account
    progression is shared by definition; a run is not.
#>
function Initialize-ClientSaves {
    param([string]$AppData, [string]$Label)

    if (-not (Test-Path -LiteralPath $SeedFrom)) {
        Write-Log "WARNING: $Label has no seed at $SeedFrom; mods may not load"
        return
    }

    $wanted = @(
        'settings.save',
        'progress.save',
        'progress.save.backup',
        'prefs.save',
        'profile.save'
    )
    $seeds = @(Get-ChildItem -LiteralPath $SeedFrom -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $wanted -contains $_.Name })
    if (@($seeds | Where-Object { $_.Name -eq 'settings.save' }).Count -eq 0) {
        Write-Log "WARNING: no settings.save under $SeedFrom; $Label may load no mods"
        return
    }
    if (@($seeds | Where-Object { $_.Name -eq 'progress.save' }).Count -eq 0) {
        Write-Log "WARNING: no progress.save under $SeedFrom; $Label may offer no singleplayer menu"
    }

    $seedRoot = (Resolve-Path -LiteralPath $SeedFrom).Path
    $target = Join-Path $AppData 'SlayTheSpire2'
    foreach ($seed in $seeds) {
        $relative = $seed.FullName.Substring($seedRoot.Length).TrimStart('\', '/')
        $destination = Join-Path $target $relative
        if (Test-Path -LiteralPath $destination) { continue }
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force
        Copy-Item -LiteralPath $seed.FullName -Destination $destination
        if ($seed.Name -eq 'settings.save') { Set-TrainingFrameRate -SettingsPath $destination }
        Write-Log "$Label seeded $relative"
    }
}

<#
    Stop the game throttling its frame rate when it is not the focused window.

    settings.save carries limit_fps_in_background, and the game honours it: a
    client that is not in the foreground -- which is every headless client, and
    every windowed client but the one you happen to have clicked on -- runs its
    main loop slowly.

    That matters here because the mod's settle logic waits on processed frames
    before it reports the state an action produced. Fewer frames per second is
    directly fewer steps per second.

    Measured on this machine over four episodes each, same client install, same
    seed pool, ~300 steps per configuration: headless with the throttle on ran
    1.166 s/step, and with it off 0.524 s/step, against 0.608 s/step windowed.
    The throttle, not the missing renderer, was the whole difference -- and it
    throttles unfocused windowed clients too, so a multi-client run pays it on
    every client but the one in the foreground.

    Only the seeded copy is edited. The real settings.save is read and never
    written, so playing the game by hand is unaffected.
#>
function Set-TrainingFrameRate {
    param([string]$SettingsPath)

    try {
        $settings = Get-Content -LiteralPath $SettingsPath -Raw | ConvertFrom-Json
    } catch {
        Write-Log "WARNING: could not parse $SettingsPath; leaving frame rate alone"
        return
    }
    if ($null -eq $settings.PSObject.Properties['limit_fps_in_background']) { return }
    if ($settings.limit_fps_in_background -eq $false) { return }

    $settings.limit_fps_in_background = $false
    $settings | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $SettingsPath -Encoding UTF8
}

function New-GameClient {
    param([int]$Index)

    $installDir = Join-Path $GameRoot ($InstanceFormat -f $Index)
    $exePath = Join-Path $installDir 'SlayTheSpire2.exe'
    if (-not (Test-Path -LiteralPath $exePath)) {
        throw "Game executable not found for client-${Index}: $exePath"
    }

    $port = $BasePort + $Index - 1
    $modPort = Get-ModPort -InstallDir $installDir
    if ($null -eq $modPort) {
        Write-Log "WARNING: client-${Index} has no readable STS2_MCP.conf; assuming port $port"
    } elseif ($modPort -ne $port) {
        # Refusing here is deliberate. A silent mismatch means the trainer
        # connects to a different client than it believes, and every metric
        # attributed to that lane is wrong with nothing failing.
        throw ("client-${Index} mod is configured for port $modPort but this " +
               "script expects $port. Edit $installDir\mods\STS2_MCP\STS2_MCP.conf " +
               "and restart, or pass a matching -BasePort.")
    }

    $client = [GameClient]::new()
    $client.Index = $Index
    $client.Port = $port
    $client.ExePath = $exePath
    $client.StdOut = Join-Path $LogDir "client-$Index.stdout.log"
    $client.StdErr = Join-Path $LogDir "client-$Index.stderr.log"
    $client.FailedChecks = 0

    if ($SharedSaves) {
        $client.AppData = $env:APPDATA
    } else {
        # This MegaDot build resolves user:// from the APPDATA environment
        # variable (verified in the binary: APPDATA is read, XDG_CONFIG_HOME is
        # not present on the Windows build). There is no --user-dir flag and
        # use_custom_user_dir is baked into the .pck, so the environment is the
        # only lever available.
        $client.AppData = Join-Path $SaveRoot "client-$Index\Roaming"
        $null = New-Item -ItemType Directory -Path $client.AppData -Force
        Initialize-ClientSaves -AppData $client.AppData -Label $client.Label()
    }

    return $client
}

# --------------------------------------------------------------------------
# Process control
# --------------------------------------------------------------------------

function Start-GameClient {
    param([GameClient]$Client)

    $gameArgs = @()
    if ($Headless) {
        # --headless is shorthand for --display-driver headless --audio-driver
        # Dummy. Spelled out so a future failure can be narrowed to one half.
        $gameArgs += @('--display-driver', 'headless', '--audio-driver', 'Dummy')
    }
    $gameArgs += $ExtraGameArgs

    Write-Log "Starting $($Client.Label()) on port $($Client.Port)$(if ($Headless) { ' (headless)' })"

    # Start-Process cannot set environment variables for the child, so set them
    # on this process around the call. Each client needs its own APPDATA and
    # they are started one at a time, so this is safe.
    $savedAppData = $env:APPDATA
    try {
        $env:APPDATA = $Client.AppData
        $startArgs = @{
            FilePath               = $Client.ExePath
            WorkingDirectory       = (Split-Path -Parent $Client.ExePath)
            RedirectStandardOutput = $Client.StdOut
            RedirectStandardError  = $Client.StdErr
            PassThru               = $true
            WindowStyle            = 'Hidden'
        }
        if ($gameArgs.Count -gt 0) { $startArgs['ArgumentList'] = $gameArgs }
        $Client.Process = Start-Process @startArgs
    } finally {
        $env:APPDATA = $savedAppData
    }

    $Client.StartedAt = Get-Date
    $Client.FailedChecks = 0
    Write-Log "$($Client.Label()) started pid=$($Client.Process.Id)"
}

function Stop-GameClient {
    param([GameClient]$Client)

    if ($null -eq $Client.Process -or $Client.Process.HasExited) {
        $Client.Process = $null
        return
    }

    Write-Log "Stopping $($Client.Label()) pid=$($Client.Process.Id)"
    try {
        # -Force because the game installs no console handler and will not
        # respond to a polite request.
        Stop-Process -Id $Client.Process.Id -Force -ErrorAction Stop
        $null = $Client.Process.WaitForExit(10000)
    } catch {
        Write-Log "Could not stop $($Client.Label()): $($_.Exception.Message)"
    }
    $Client.Process = $null
}

function Test-ClientHealth {
    param([GameClient]$Client)

    try {
        $null = Invoke-WebRequest -Uri $Client.HealthUrl() `
            -TimeoutSec $HealthTimeoutSeconds -UseBasicParsing
        return $true
    } catch {
        return $false
    }
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

$null = New-Item -ItemType Directory -Path $LogDir -Force

Write-Log "Game root: $GameRoot"
Write-Log ("Mode: {0}, saves: {1}" -f
    $(if ($Headless) { 'headless' } else { 'windowed' }),
    $(if ($SharedSaves) { 'SHARED (clients will overwrite each other)' } else { "isolated under $SaveRoot" }))

$wanted = if ($PSBoundParameters.ContainsKey('Indices')) { $Indices } else { 1..$Count }
$clients = @($wanted | ForEach-Object { New-GameClient -Index $_ })

if ($ValidateOnly) {
    Write-Log 'Validation only -- nothing will be started.'
    foreach ($client in $clients) {
        Write-Host ("  {0}  port={1}  exe={2}" -f $client.Label(), $client.Port, $client.ExePath)
        Write-Host ("      APPDATA -> {0}" -f $client.AppData)
        Write-Host ("      health  -> {0}" -f $client.HealthUrl())
    }
    # @() everywhere: a one-element pipeline unrolls to a scalar, which has no
    # .Count, so these checks would crash on a single client.
    $ports = @($clients | ForEach-Object { $_.Port })
    if (@($ports | Select-Object -Unique).Count -ne $ports.Count) {
        throw 'Two clients resolved to the same port.'
    }
    $saves = @($clients | ForEach-Object { $_.AppData })
    if (-not $SharedSaves -and @($saves | Select-Object -Unique).Count -ne $saves.Count) {
        throw 'Two clients resolved to the same save directory.'
    }
    Write-Log 'Validation passed.'
    return
}

# Register cleanup before starting anything, so an early failure does not leave
# orphans. The batch scripts had no equivalent and leaked processes on Ctrl+C.
try {
    foreach ($client in $clients) { Start-GameClient -Client $client }

    Write-Log "All $($clients.Count) clients started. Ctrl+C to stop them all."

    while ($true) {
        Start-Sleep -Seconds $CheckIntervalSeconds

        foreach ($client in $clients) {
            if ($null -eq $client.Process -or $client.Process.HasExited) {
                Write-Log "$($client.Label()) is not running"
                Start-Sleep -Seconds $RestartCooldownSeconds
                Start-GameClient -Client $client
                continue
            }

            # A client that has not finished loading yet is not unhealthy.
            $age = (Get-Date) - $client.StartedAt
            if ($age.TotalSeconds -lt $StartupGraceSeconds) { continue }

            if (Test-ClientHealth -Client $client) {
                if ($client.FailedChecks -gt 0) {
                    Write-Log "$($client.Label()) is responsive again"
                }
                $client.FailedChecks = 0
                continue
            }

            $client.FailedChecks++
            Write-Log ("$($client.Label()) health check failed " +
                       "($($client.FailedChecks)/$MaxFailedHealthChecks) $($client.HealthUrl())")

            if ($client.FailedChecks -ge $MaxFailedHealthChecks) {
                Write-Log "Restarting $($client.Label())"
                Stop-GameClient -Client $client
                Start-Sleep -Seconds $RestartCooldownSeconds
                Start-GameClient -Client $client
            }
        }
    }
} finally {
    Write-Log 'Shutting down all clients'
    foreach ($client in $clients) { Stop-GameClient -Client $client }
}

# Training

STS2RL trains one structured candidate-PPO agent against a local STS2MCP
single-player server. The runtime runs complete episodes serially, writes an
auditable JSONL log, mirrors scalar metrics to TensorBoard, and saves resumable
checkpoints at clean episode boundaries.

## Start a run

Start STS2 and STS2MCP first, then run:

```powershell
uv run python scripts/train.py `
  --run-dir runs/ironclad-001 `
  --total-episodes 100 `
  --character 0 `
  --device cpu
```

The run directory must be empty for a new experiment. This prevents an
accidental invocation from overwriting another model or mixing unrelated
metrics. Run `python scripts/train.py --help` for all encoder, PPO, environment,
and checkpoint options.

`--total-episodes` is a cumulative target. It means "stop when this run has
completed this many episodes," including episodes restored from a checkpoint.

## Resume

Resume the checkpoint named by `checkpoints/latest.json`:

```powershell
uv run python scripts/train.py `
  --run-dir runs/ironclad-001 `
  --resume latest `
  --total-episodes 500
```

The checkpoint restores the encoder, optimizer, completed episodes,
environment-step and PPO-update counters, and CPU/CUDA random-number state.
Model-defining encoder, PPO, character, and game-mode settings cannot change
during resume. Operational settings such as the cumulative episode target,
checkpoint frequency, server URL, timeout, device, and TensorBoard flush period
may change.

Resume starts a fresh game episode. It does not restore a battle or other STS2
screen that was open when the process stopped. By default, reset therefore
rejects an existing active run rather than silently treating its middle as a
new episode.

## Run artifacts

```text
runs/ironclad-001/
|-- config.json
|-- metrics.jsonl
|-- tensorboard/
|   `-- events.out.tfevents...
`-- checkpoints/
    |-- episode_000010.pt
    |-- episode_000020.pt
    `-- latest.json
```

`config.json` records the initial effective configuration. Each checkpoint also
contains its effective configuration and a SHA-256 vocabulary fingerprint. A
checkpoint is rejected when categorical embedding rows would no longer have
the same meaning.

Checkpoint publication is atomic: the runtime writes a temporary file, moves
it into place, and only then updates `latest.json`. TensorBoard event files stay
outside the checkpoint; the checkpoint stores the log directory and last
global step needed to continue them.

## TensorBoard

TensorBoard logging is enabled by default:

```powershell
tensorboard --logdir runs/ironclad-001/tensorboard
```

All scalar series use cumulative environment steps as their global step. The
runtime records episode reward, length, floor, action errors, termination,
duration, PPO losses, entropy, gradient norm, rollout size, completed episodes,
and update count.

On resume, the writer uses the checkpoint step to hide event records produced
after that checkpoint. The JSONL log is atomically trimmed to the same boundary.
Disable TensorBoard for a new run with `--no-tensorboard`; JSONL remains active.

## Seeds and devices

`--torch-seed` controls model initialization and policy sampling. A checkpoint
restores saved RNG state instead of reseeding. STS2 run seeds are only meaningful
for custom or daily runs, so standard mode rejects `--run-seed`.

CPU is the default. Use `--device cuda` or a specific CUDA device only when it
is available. A checkpoint can be mapped to a different runtime device; CUDA
RNG state is restored only when resuming on CUDA.

## Interrupts and failures

Ctrl+C discards an incomplete PPO rollout, flushes JSONL and TensorBoard, and
saves an `interrupted_step_*.pt` checkpoint containing all updates completed
before the interruption. The command exits with status 130.

Unexpected tokenizer, environment, or observation errors follow the same safe
cleanup path, save `recovery_step_*.pt` when possible, and are then re-raised.
They are not retried indefinitely because repeated schema errors require
diagnosis rather than another HTTP request.

## Current limits

The first runtime intentionally has one serial environment and computes PPO one
transition at a time. It does not yet provide exact mid-episode game restore,
padded minibatches, parallel environments, reward redesign, automatic fixture
capture, checkpoint evaluation, or best-model selection.

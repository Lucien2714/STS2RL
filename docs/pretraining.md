# Pretraining (Behavioral Cloning)

Bootstrapping the battle network from human play before RL gives the agent a
sensible starting policy instead of learning from scratch. This addresses the
failure mode where a from-scratch DQN collapses into degenerate behavior (an
inflated `end_turn` Q-value, ending its own turns to escape unwinnable states).
The workflow is: record human play, clone it supervised, then fine-tune with RL.

## 1. Record human play (STS2MCP mod)

The STS2MCP mod records the local player's gameplay when recording is enabled.
Turn it on either way:

- **In game**: Settings → Mods → **Record Human Actions**.
- **Config file**: set `"record_human_actions": true` in `STS2_MCP.conf` (next to
  the mod DLL).

Play the game normally. The mod writes one decision per line to
`<game>/mods/recordings/human_<timestamp>.jsonl`:

```json
{"ts": 1.23, "state_type": "monster", "state": { ... }, "action": {"type": "play_card", "card_index": 0, "target": "JAW_WORM_0"}}
```

`state` is the same payload the HTTP API returns, and `action` uses the same
schema the API accepts, so recordings encode identically to live RL transitions.
Actions issued by the API/agent are **not** recorded, so evaluating or training
the agent against a running client never pollutes the human dataset.

Currently the recorder captures combat actions (`play_card`, `end_turn`,
`use_potion`, `discard_potion`) — the battle agent's decision space, which is the
pretraining target. See the mod's `McpMod.HumanRecording.cs` to extend it.

## 2. Pretrain (STS2RL)

```bash
uv run sts2rl-pretrain --recordings /path/to/recordings --battle-agent DQN --epochs 20
```

What it does:

- Reads the JSONL recordings (a file, directory, or glob).
- For each `(state, action)`, enumerates the agent's legal candidate actions,
  scores them, and trains with softmax cross-entropy to give the candidate the
  human chose the highest probability. Because the DQN and PPO networks both
  score `state + candidate action → scalar`, the same loss pretrains either; new
  candidate-scoring algorithms work by registering in
  `orchestrator.BATTLE_AGENT_TYPES`.
- Each recorded action is matched to one candidate via `action_key`. The run
  reports the **match rate** (how many recordings mapped onto a legal candidate)
  and **top-1 accuracy** on a held-out split. A low match rate means the recorded
  schema and the agent's candidate generation have drifted apart.
- Saves a checkpoint in the agent's normal format (default:
  `checkpoints/battleAgent/<TYPE>/battleagent_latest.pt`), writing a low
  `epsilon` so RL exploits the cloned policy rather than overwriting it.

Useful flags: `--out`, `--lr`, `--batch-size`, `--val-split`, `--epsilon`,
`--limit`, `--seed`. See `uv run sts2rl-pretrain --help`.

## 3. Fine-tune with RL

`sts2rl-train` loads the latest checkpoint for the agent type, so once the
pretrained checkpoint is in place RL continues from it:

```bash
uv run sts2rl-train --client-port 15526
```

Spot-check that early behavior is non-degenerate (it should not immediately spam
`end_turn`). The pretrained checkpoint uses the same `action_schema`, so a schema
mismatch on load means the recordings were produced against an incompatible
agent version — re-record or pretrain with the matching agent.

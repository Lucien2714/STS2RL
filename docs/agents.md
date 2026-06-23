# Agents

Agent interfaces live in `sts2rl.agents.base`.

`ScreenAgent` is the minimal screen-control interface. `TrainableScreenAgent`,
`MapAgent`, and `EventAgent` specialize it for important screen families.
(`BattleAgent` is a backward-compatible alias of `TrainableScreenAgent` — the
interface began life as the battle-agent base and is not battle-specific.)

`TrainableScreenAgent` adds optional learning and persistence hooks:

```python
choose_action(state, training=False)
observe(transition)
save(path)
load(path)
```

Use `choose_action` at the agent boundary. Use `forward` only for neural network
modules. Use `env.step(action)` for environment transitions.

Current concrete agents:

- `DQNCandidateAgent` (`sts2rl.agents.candidate_dqn_agent`)
- `PPOCandidateAgent` (`sts2rl.agents.candidate_ppo_agent`)
- `MapPolicy` / `RuleBasedMapAgent`
- `EventPolicy` / `RuleBasedEventAgent`
- rule-based reward, rest, shop, and default policies

`DQNCandidateAgent` / `PPOCandidateAgent` are the canonical names: they score a
set of legal action candidates for *one screen* and compose a screen-specific
encoder (see `encoders/`). A bare instance defaults to the battle encoder and is
the battle agent; pass `encoder=<ScreenEncoder>()` to drive a non-battle screen
(map, reward, shop, rest, event). The battle-named aliases `BattleDQNAgent` /
`DQNBattleAgent` / `BattlePPOAgent` / `PPOBattleAgent` remain available — the
`sts2rl.agents.battle.dqn_agent` / `...ppo_agent` modules are thin re-export
shims for backward compatibility.

Each non-battle screen also has a dedicated, importable agent class per algorithm
that binds its encoder — `MapDQNAgent` / `MapPPOAgent` in
`sts2rl.agents.map.agent`, and likewise `Reward*`, `Shop*`, `Rest*`, `Event*` in
each `sts2rl.agents.<screen>.agent`. These thin subclasses are the discoverable
type and extension point for per-screen behavior; the encoder still owns all
screen-specific encoding and candidate logic. The orchestrator registry
`SCREEN_AGENTS` (`screen → {"DQN", "PPO"} → class`) maps them, and the factories
`create_screen_agent(screen, type)` / `create_screen_agents(type)` build them.
Each screen agent gets its own schema (`map_dqn_v1`, `reward_ppo_v1`, …), model,
optimizer, and checkpoint file (`checkpoints/<screen>Agent/<TYPE>/...`), so screens
train, save, load, pretrain, and evaluate independently of the battle agent and of
each other. Screens with no registered/loaded agent fall back to their rule-based
policy.

The top-level `Agent` uses `DQNCandidateAgent` (battle encoder) by default, but
accepts an explicit battle agent type or instance for experiments. Battle routing
includes combat screens, `hand_select`, and `card_select` overlays while
`raw_state["in_battle"]` is true:

```python
from sts2rl.agents.orchestrator import Agent
from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent

agent = Agent(battle_agent_type="PPO")
agent = Agent(battle_agent=PPOCandidateAgent())  # BattlePPOAgent alias also works
```

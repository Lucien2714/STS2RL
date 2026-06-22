# Agents

Agent interfaces live in `sts2rl.agents.base`.

`ScreenAgent` is the minimal screen-control interface. `BattleAgent`,
`MapAgent`, and `EventAgent` specialize it for important screen families.

`BattleAgent` adds optional learning and persistence hooks:

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

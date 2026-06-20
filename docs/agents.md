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

- `BattleDQNAgent` / `DQNBattleAgent`
- `BattlePPOAgent` / `PPOBattleAgent`
- `MapPolicy` / `RuleBasedMapAgent`
- `EventPolicy` / `RuleBasedEventAgent`
- rule-based reward, rest, shop, and default policies

The top-level `Agent` uses `BattleDQNAgent` by default, but accepts an explicit
battle agent type or instance for experiments. Battle routing includes combat
screens, `hand_select`, and `card_select` overlays while `raw_state["in_battle"]`
is true:

```python
from sts2rl.agents.orchestrator import Agent
from sts2rl.agents.battle import BattlePPOAgent

agent = Agent(battle_agent_type="PPO")
agent = Agent(battle_agent=BattlePPOAgent())
```

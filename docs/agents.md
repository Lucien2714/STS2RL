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
- `MapPolicy` / `RuleBasedMapAgent`
- `EventPolicy` / `RuleBasedEventAgent`
- rule-based reward, rest, shop, and default policies


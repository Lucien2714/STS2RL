# Agents

Agent interfaces live in `sts2rl.agents.base`.

`ScreenAgent` is the minimal screen-control interface. `TrainableScreenAgent`,
`MapAgent`, and `EventAgent` specialize it for important screen families.

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

`DQNCandidateAgent` / `PPOCandidateAgent` are *algorithm* classes: each owns only
its update rule, optimizer, buffers, and checkpoint IO, and composes three
swappable components (see [ADR-0007](adr/0007-action-space-policy-algorithm-split.md)):

- an **action space** (`action_spaces/`) — torch-free legal-action enumeration;
- an **encoder** (`encoders/`) — model-free state/action featurization;
- a **policy module** (`models/policies.py`) — the scoring `nn.Module`
  (`CandidateQNetwork` / `CandidatePPOPolicy` by default; pass `policy=` to swap
  architectures).

A bare instance defaults to the battle space/encoder and is the battle agent. A
non-battle screen agent is a *configuration*, not a subclass: the orchestrator
registry `SCREEN_SPECS` (`screen → ScreenSpec(action_space, encoder)`) maps each
screen (map, reward, shop, rest, event) to its components, and the factories
`create_screen_agent(screen, type, **kwargs)` / `create_screen_agents(type)`
bind them to the chosen algorithm. Each screen agent gets its own schema
(`map_dqn_v1`, `reward_ppo_v1`, …), model, optimizer, and checkpoint file
(`checkpoints/<screen>Agent/<TYPE>/...`), so screens train, save, load, pretrain,
and evaluate independently of the battle agent and of each other. Screens with no
registered/loaded agent fall back to their rule-based policy.

The top-level `Agent` uses `DQNCandidateAgent` (battle encoder) by default, but
accepts an explicit battle agent type or instance for experiments. Battle routing
includes combat screens, `hand_select`, and `card_select` overlays while
`raw_state["in_battle"]` is true:

```python
from sts2rl.agents.orchestrator import Agent
from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent

agent = Agent(battle_agent_type="PPO")
agent = Agent(battle_agent=PPOCandidateAgent())
```

# Agents

The first rebuilt agent uses PPO over a dynamic set of complete structured
actions. There is no global discrete action ID. For every raw state,
`LegalActionProvider` creates candidates such as:

```python
[
    GameAction("play_card", card_index=0, target="ENEMY_0"),
    GameAction("use_potion", slot=0, target="ENEMY_0"),
    GameAction("end_turn"),
]
```

`CandidatePPOAgent` scores only those candidates and samples from the resulting
categorical distribution. Its rollout stores the candidate features used at
sampling time, selected index, old log probability, value, reward, and terminal
flag. This preserves the PPO probability ratio even though the action set can
change after every step.

```python
from sts2rl.agents import CandidatePPOAgent, EpisodeRunner
from sts2rl.env import GameEnv, ResetSpec

env = GameEnv()
agent = CandidatePPOAgent()
runner = EpisodeRunner(env, agent)
result = runner.run(ResetSpec(character=0))
```

The initial `HashingFeatureEncoder` makes the loop runnable without changing the
retained `sts2rl.encoder.StateEncoder`. It is a baseline boundary: a future
domain encoder can replace it by providing `state_dim`, `action_dim`,
`encode_state()`, and `encode_action()`.

The legal-action provider covers combat, in-combat selection, rewards, map,
events, rest sites, shops, treasure, card/bundle/relic overlays, and the Crystal
Sphere. It returns no guessed action for `unknown` or unhandled `overlay`
states. Pre-run menus also remain the responsibility of `ResetController`, not
the learning agent. The runner refreshes short-lived states a bounded number of
times and then raises `NoLegalActionsError` with the raw state for diagnosis.

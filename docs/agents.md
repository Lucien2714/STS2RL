# Agents

The structured token data contract used by the trainable encoder is documented
in [Structured Game Encoding](encoding.md). PPO trains that encoder end to end;
there is no intermediate flat feature-vector interface.

The first rebuilt agent uses PPO over a dynamic set of complete structured
actions. There is no global discrete action ID. For every observation's raw
state,
`LegalActionProvider` creates candidates such as:

```python
[
    GameAction("play_card", card_index=0, target="ENEMY_0"),
    GameAction("use_potion", slot=0, target="ENEMY_0"),
    GameAction("end_turn"),
]
```

`CandidatePPOAgent` tokenizes and scores only those candidates, then samples
from the resulting categorical distribution. Its rollout stores the complete
CPU `TokenizedDecision`, the next `GameObservation`, the selected index, the old
log probability, the old value, the reward, and the terminal flag. Only the
final step's next state is ever tokenized, and only when bootstrapping, so the
common case pays nothing for it. PPO updates rerun `GameEncoder` from those
snapshots, so gradients reach
the categorical embeddings, entity transformer, map DAG encoder, action
encoder, and policy/value heads while the candidate identity remains stable.

```python
from sts2rl.agents import CandidatePPOAgent, EpisodeRunner
from sts2rl.encoder import EncoderConfig, GameEncoder, GameTokenizer, GameVocabulary
from sts2rl.env import GameEnv, ResetSpec

env = GameEnv()
vocabulary = GameVocabulary.from_bundled_data()
tokenizer = GameTokenizer(vocabulary)
encoder = GameEncoder(vocabulary, EncoderConfig())
agent = CandidatePPOAgent(tokenizer=tokenizer, game_encoder=encoder)
runner = EpisodeRunner(env, agent)
result = runner.run(ResetSpec(character=0))
```

The tokenizer and encoder are explicit constructor dependencies. This keeps the
vocabulary, token schema, model dimensions, device placement, and optimizer
parameters visible to the training entry point.

The Agent lifecycle now receives `GameObservation` rather than a bare raw
dictionary:

```text
reset(initial_observation)
choose_action(observation)
observe(Transition[GameObservation])
finish_episode(final_observation, truncated)
```

A state offering exactly one candidate is executed directly and never enters
the rollout: its log probability is always zero and its ratio always one, so it
carries no policy gradient while still diluting the advantage statistics of
genuine decisions. Its reward is folded into the preceding recorded decision,
so the return of every trained step still matches what the environment paid.

The default long-horizon return settings are `gamma=0.999` and
`gae_lambda=0.98`. A terminal step bootstraps with zero; a non-terminal rollout
boundary or truncated episode bootstraps from the stored next observation.
Updates run over shuffled minibatches (`minibatch_size`, default 32), which also
bounds how many autograd graphs are live at once. Training samples
stochastically, while `agent.eval()` selects the highest-logit candidate
deterministically and does not collect rollout entries.

The training runtime drains metrics for every completed PPO update instead of
only reading `last_update`. At a clean episode boundary the agent serializes its
encoder and optimizer; the lifetime counters belong to `TrainingState` and are
stored once by the checkpoint rather than duplicated here. Incomplete pending
actions and rollouts must be observed, updated, or explicitly aborted before
checkpointing.

The legal-action provider covers combat, in-combat selection, rewards, map,
events, rest sites, shops, treasure, card/bundle/relic overlays, and the Crystal
Sphere. It returns no guessed action for `unknown` or unhandled `overlay`
states. Pre-run menus also remain the responsibility of `ResetController`, not
the learning agent. The runner re-reads short-lived states a bounded number of
times with exponential backoff — combat outside the play phase legally exposes
no action, and the server needs time to settle — and then truncates the episode
rather than raising through the trainer and ending the whole run.

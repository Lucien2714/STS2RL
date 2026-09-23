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
log probability, the old value in raw reward units, the reward, and separate
terminal and episode-boundary flags. Next states are tokenized at nonterminal
rollout boundaries and episode truncations for bootstrapping. PPO updates rerun `GameEncoder` from those
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

A state offering exactly one candidate is recorded as a critic-only transition.
Its log probability is zero; it is excluded from policy loss, entropy, and
policy advantage normalization. Keeping the transition preserves the time of
its reward, the terminal flag, and the per-environment-step discount and GAE
trace. Rewards are never moved onto preceding or following decisions. The
default rollout size of 256 now counts all recorded environment transitions,
including forced actions; it previously counted only genuine choices.

The default long-horizon return settings are `gamma=0.999` and
`gae_lambda=0.98`. A terminal step bootstraps with zero; a non-terminal rollout
boundary or truncated episode bootstraps from its own final observation. A
truncation cuts the GAE trace without zeroing that bootstrap, so rewards from
the next episode cannot flow backward into it. Sampling values and truncation
bootstraps are captured in raw reward units under the model lock; subsequent
return-scale updates cannot change the meaning of an in-flight value. These
changes preserve the model and checkpoint format. They do not make the parallel
collector synchronous: another lane may still finish an action sampled before
the latest update, with its original log probability and raw value retained.
Updates run over shuffled minibatches (`minibatch_size`, default 32), which also
bounds how many autograd graphs are live at once. Training samples
stochastically, while `agent.eval()` selects the highest-logit candidate
deterministically and does not collect rollout entries.

The training runtime drains metrics for every completed PPO update instead of
only reading `last_update`. Checkpointing takes a locked snapshot of the encoder,
optimizer, and return scale without flushing an incomplete rollout. Pending
actions and rollout entries are not serialized. Lifetime counters belong to
`TrainingState` and are stored once by the checkpoint. Loading requires an empty
rollout and no pending actions.

The legal-action provider covers combat, in-combat selection, rewards, map,
events, rest sites, shops, treasure, card/bundle/relic overlays, and the Crystal
Sphere. It returns no guessed action for `unknown` or unhandled `overlay`
states. Pre-run menus also remain the responsibility of `ResetController`, not
the learning agent. The runner re-reads short-lived states a bounded number of
times with exponential backoff — combat outside the play phase legally exposes
no action, and the server needs time to settle — and then truncates the episode
rather than raising through the trainer and ending the whole run.

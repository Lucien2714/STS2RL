"""Tests for the PPO battle agent."""

from sts2rl.agents.battle.ppo_agent import BattlePPOAgent, PPOBattleAgent
from sts2rl.agents.orchestrator import Agent


def playable_battle_state(enemy_hp: int = 10) -> dict:
    """Build a small battle state with at least two legal actions."""
    return {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [
                {"entity_id": "ENEMY_0", "hp": enemy_hp, "max_hp": 10},
            ],
        },
        "player": {
            "energy": 3,
            "max_energy": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "STRIKE_IRONCLAD",
                    "type": "Attack",
                    "cost": "1",
                    "target_type": "Enemy",
                    "can_play": True,
                },
            ],
            "potions": [],
            "status": [],
            "relics": [],
        },
    }


def test_ppo_alias_exports_agent_class():
    """The short PPO alias should match the exported agent."""
    assert PPOBattleAgent is BattlePPOAgent


def test_ppo_choose_action_returns_legal_candidate_with_probability():
    """PPO action selection should sample from legal candidate actions."""
    agent = BattlePPOAgent(rollout_steps=2, hidden_size=32)
    raw_state = playable_battle_state()

    action = agent.choose_action(raw_state, training=True)
    action_keys = {
        candidate["action_key"] for candidate in agent.valid_action_candidates(raw_state)
    }

    assert agent.action_key(action, raw_state) in action_keys
    assert agent.last_action_selection["method"] == "ppo_sample"
    assert agent.last_action_selection["prob"] is not None
    assert agent.last_action_selection["value"] is not None


def test_ppo_train_step_updates_after_rollout_and_clears_buffer():
    """A full rollout should run one clipped PPO update."""
    agent = BattlePPOAgent(
        rollout_steps=2,
        minibatch_size=1,
        update_epochs=1,
        hidden_size=32,
    )
    state = playable_battle_state()
    next_state = playable_battle_state(enemy_hp=5)

    for reward, done in [(1.0, False), (2.0, True)]:
        action = agent.choose_action(state, training=True)
        agent.remember(
            agent.encode_state(state),
            agent.encode_action(state, action),
            reward,
            agent.encode_state(next_state),
            done,
            [] if done else agent.candidate_action_vectors(next_state),
        )

    loss = agent.train_step()

    assert isinstance(loss, float)
    assert agent.learn_steps == 1
    assert agent.buffered_steps() == 0


def test_ppo_checkpoint_round_trip(tmp_path):
    """PPO checkpoints should load into a fresh PPO agent."""
    path = tmp_path / "battle_ppo.pt"
    agent = BattlePPOAgent(rollout_steps=2, hidden_size=32)
    agent.trained_steps = 3
    agent.learn_steps = 1

    agent.save(str(path))
    loaded = BattlePPOAgent(rollout_steps=2, hidden_size=32)
    loaded.load(str(path))

    assert loaded.trained_steps == 3
    assert loaded.learn_steps == 1


def test_orchestrator_accepts_ppo_battle_agent():
    """The top-level policy router can be constructed with PPO."""
    battle_agent = BattlePPOAgent(rollout_steps=2, hidden_size=32)
    agent = Agent(battle_agent=battle_agent)

    action = agent.choose_action(
        {
            "screen_type": "monster",
            "raw_state": playable_battle_state(),
        }
    )

    assert agent.battle_agent is battle_agent
    assert action["type"] in {"end_turn", "play_card"}


def test_orchestrator_creates_ppo_from_agent_type():
    """The top-level policy router can build PPO by type name."""
    agent = Agent(battle_agent_type="PPO")

    assert isinstance(agent.battle_agent, BattlePPOAgent)


def test_ppo_collectors_isolate_pending_transitions():
    """Interleaved collectors must not clobber each other's on-policy pending data."""
    agent = BattlePPOAgent(rollout_steps=8, hidden_size=32)
    collector_a = agent.new_rollout()
    collector_b = agent.new_rollout()
    state_a = playable_battle_state()
    state_b = playable_battle_state(enemy_hp=7)

    # A chooses, then B chooses (this would overwrite a single shared pending slot),
    # then each remembers in interleaved order.
    action_a = collector_a.choose_action(state_a, training=True)
    action_b = collector_b.choose_action(state_b, training=True)
    collector_a.remember(
        agent.encode_state(state_a),
        agent.encode_action(state_a, action_a),
        1.0,
        agent.encode_state(state_a),
        True,
        [],
    )
    collector_b.remember(
        agent.encode_state(state_b),
        agent.encode_action(state_b, action_b),
        1.0,
        agent.encode_state(state_b),
        True,
        [],
    )

    # Each kept its own transition with the full candidate set (end_turn + play_card),
    # not the single-candidate fallback that the old shared-slot clobber produced.
    assert len(collector_a.buffer) == 1
    assert len(collector_b.buffer) == 1
    assert len(collector_a.buffer[0]["candidate_vectors"]) >= 2
    assert len(collector_b.buffer[0]["candidate_vectors"]) >= 2


def test_ppo_update_groups_trajectories_and_clears_buffers():
    """An update over two collectors runs one PPO step and clears every buffer."""
    agent = BattlePPOAgent(
        rollout_steps=4,
        minibatch_size=2,
        update_epochs=1,
        hidden_size=32,
    )
    collectors = [agent.new_rollout(), agent.new_rollout()]

    for collector in collectors:
        state = playable_battle_state()
        next_state = playable_battle_state(enemy_hp=5)
        for reward, done in [(1.0, False), (2.0, True)]:
            action = collector.choose_action(state, training=True)
            collector.remember(
                agent.encode_state(state),
                agent.encode_action(state, action),
                reward,
                agent.encode_state(next_state),
                done,
                [] if done else agent.candidate_action_vectors(next_state),
            )

    assert agent.buffered_steps() == 4  # two trajectory segments of length 2

    loss = agent.train_step()

    assert isinstance(loss, float)
    assert agent.learn_steps == 1
    assert agent.buffered_steps() == 0
    assert all(collector.buffer == [] for collector in collectors)

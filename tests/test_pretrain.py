"""Tests for behavioral-cloning pretraining and the recordings reader."""

import json

from sts2rl.agents.battle.dqn_agent import BattleDQNAgent
from sts2rl.data.recordings import iter_recordings, load_recordings
from sts2rl.training.pretrain import build_examples, pretrain


def battle_state(enemy_hp: int = 10) -> dict:
    """Battle state with a targeted attack, a self potion, and end_turn legal."""
    return {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [{"entity_id": "ENEMY_0", "hp": enemy_hp, "max_hp": 10}],
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
            "potions": [
                {"slot": 0, "id": "BLOCK_POTION", "target_type": "None", "can_use_in_combat": True},
            ],
            "status": [],
            "relics": [],
        },
    }


def sample_recordings() -> list[tuple[dict, dict]]:
    """Recorded (state, action) pairs in the mod's API action schema."""
    return [
        (battle_state(), {"type": "play_card", "card_index": 0, "target": "ENEMY_0"}),
        (battle_state(enemy_hp=6), {"type": "end_turn"}),
        (battle_state(), {"type": "use_potion", "slot": 0}),
    ]


def write_jsonl(path, samples, extra_lines=()):
    """Write samples in the mod's JSONL recording format."""
    with path.open("w", encoding="utf-8") as handle:
        for state, action in samples:
            handle.write(json.dumps({
                "ts": 0.0,
                "state_type": state["state_type"],
                "state": state,
                "action": action,
            }) + "\n")
        for line in extra_lines:
            handle.write(line + "\n")


def test_iter_recordings_reads_and_filters(tmp_path):
    """The reader yields (state, action) pairs and skips blank/malformed lines."""
    path = tmp_path / "session.jsonl"
    write_jsonl(path, sample_recordings(), extra_lines=["", "{ not json"])

    pairs = load_recordings(path)
    assert len(pairs) == 3
    assert pairs[0][1]["type"] == "play_card"

    # Action-type filtering keeps only requested types.
    only_play = list(iter_recordings(path, action_types={"play_card"}))
    assert len(only_play) == 1
    assert only_play[0][1]["type"] == "play_card"


def test_iter_recordings_reads_directory(tmp_path):
    """A directory expands to its JSONL files."""
    write_jsonl(tmp_path / "a.jsonl", sample_recordings()[:1])
    write_jsonl(tmp_path / "b.jsonl", sample_recordings()[1:])
    assert len(load_recordings(tmp_path)) == 3


def test_recorded_actions_match_exactly_one_candidate():
    """Each recorded action must map onto exactly one legal candidate (schema contract)."""
    agent = BattleDQNAgent(hidden_size=16)
    for raw_state, action in sample_recordings():
        target_key = agent.action_key(action, raw_state)
        candidate_keys = [c["action_key"] for c in agent.valid_action_candidates(raw_state)]
        assert candidate_keys.count(target_key) == 1, (action, candidate_keys)


def test_build_examples_matches_all_recordings():
    """build_examples should match every well-formed recording and report the rate."""
    agent = BattleDQNAgent(hidden_size=16)
    examples, stats = build_examples(agent, sample_recordings())

    assert stats.total == 3
    assert stats.matched == 3
    assert stats.match_rate == 1.0
    assert all(0 <= ex.target_index < ex.state_action.shape[0] for ex in examples)
    assert examples[0].state_action.shape[1] == agent.model_input_size


def test_build_examples_counts_unmatched_action():
    """An action that is not among the candidates is counted as no_match, not crashed."""
    agent = BattleDQNAgent(hidden_size=16)
    # slot 5 has no potion, so use_potion:5 produces no matching candidate.
    bad = [(battle_state(), {"type": "use_potion", "slot": 5})]
    examples, stats = build_examples(agent, bad)

    assert examples == []
    assert stats.no_match == 1


def test_pretrain_runs_and_checkpoint_round_trips(tmp_path):
    """BC training runs end-to-end and saves a checkpoint a fresh agent can load."""
    agent = BattleDQNAgent(hidden_size=16)
    examples, _ = build_examples(agent, sample_recordings() * 8)

    pretrain(agent, examples, epochs=3, batch_size=4, val_split=0.25)
    agent.epsilon = agent.epsilon_min

    path = tmp_path / "pretrained.pt"
    agent.save(str(path))

    loaded = BattleDQNAgent(hidden_size=16)
    loaded.load(str(path))
    assert loaded.epsilon == agent.epsilon_min

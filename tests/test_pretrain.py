"""Tests for behavioral-cloning pretraining and the recordings reader."""

import json
from pathlib import Path

import pytest

from sts2rl.agents.battle.dqn_agent import BattleDQNAgent
from sts2rl.agents.map.agent import MapDQNAgent
from sts2rl.agents.orchestrator import (
    SCREEN_NAMES,
    is_battle_policy_state,
    screen_name_for_state,
)
from sts2rl.data.recordings import iter_recordings, load_recordings
from sts2rl.training.pretrain import (
    PretrainTarget,
    bucket_samples,
    build_examples,
    main,
    parse_args,
    pretrain,
    resolve_target_names,
    screen_samples,
)


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
            handle.write(
                json.dumps(
                    {
                        "ts": 0.0,
                        "state_type": state["state_type"],
                        "state": state,
                        "action": action,
                    }
                )
                + "\n"
            )
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


def forced_end_turn_state() -> dict:
    """Battle state where end_turn is the only legal action (no plays, no potions)."""
    return {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [{"entity_id": "ENEMY_0", "hp": 10, "max_hp": 10}],
        },
        "player": {
            "energy": 0,
            "max_energy": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "STRIKE_IRONCLAD",
                    "type": "Attack",
                    "cost": "1",
                    "target_type": "Enemy",
                    "can_play": False,
                },
            ],
            "potions": [],
            "status": [],
            "relics": [],
        },
    }


def test_build_examples_skips_single_candidate_states():
    """Forced states (only end_turn legal) teach nothing, so they are skipped."""
    agent = BattleDQNAgent(hidden_size=16)
    state = forced_end_turn_state()
    # Sanity check: the encoder really exposes only end_turn here.
    assert [c["action_key"] for c in agent.valid_action_candidates(state)] == ["end_turn"]

    examples, stats = build_examples(agent, [(state, {"type": "end_turn"})])
    assert examples == []
    assert stats.single_candidate == 1
    assert stats.matched == 0
    assert stats.no_match == 0


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


def map_state() -> dict:
    """Map state with two legal next-node choices."""
    return {
        "state_type": "map",
        "map": {
            "next_options": [
                {"index": 0, "type": "Monster", "col": 2, "row": 3},
                {"index": 1, "type": "Shop", "col": 3, "row": 3},
            ]
        },
        "player": {"hp": 40, "max_hp": 80, "gold": 99},
        "run": {"floor": 3, "act": 1},
    }


def map_recordings() -> list[tuple[dict, dict]]:
    """Recorded map choices in the mod's API action schema."""
    return [
        (map_state(), {"type": "choose_map_node", "index": 0}),
        (map_state(), {"type": "choose_map_node", "index": 1}),
    ]


def test_build_examples_matches_screen_recordings():
    """A screen agent matches its recordings via the shared candidate interface."""
    agent = MapDQNAgent(hidden_size=16)
    examples, stats = build_examples(agent, map_recordings())

    assert stats.total == 2
    assert stats.matched == 2
    assert examples[0].state_action.shape[1] == agent.model_input_size


def test_screen_samples_filters_foreign_states(tmp_path):
    """screen_samples keeps only states the screen controls (drops other screens)."""
    reward_record = (
        {
            "state_type": "rewards",
            "rewards": {"items": [{"index": 0, "type": "card"}], "can_proceed": True},
            "player": {"hp": 40, "max_hp": 80, "gold": 99},
            "run": {"floor": 3, "act": 1},
        },
        {"type": "proceed"},
    )
    path = tmp_path / "run.jsonl"
    write_jsonl(path, [*map_recordings(), reward_record])

    kept = list(screen_samples([path], "map"))
    assert len(kept) == 2
    assert all(action["type"] == "choose_map_node" for _, action in kept)


def test_pretrain_screen_agent_round_trips(tmp_path):
    """Screen-agent BC runs end-to-end and the checkpoint loads with its own schema."""
    agent = MapDQNAgent(hidden_size=16)
    examples, _ = build_examples(agent, map_recordings() * 8)

    pretrain(agent, examples, epochs=2, batch_size=4, val_split=0.25)

    path = tmp_path / "mapagent.pt"
    agent.save(str(path))

    loaded = MapDQNAgent(hidden_size=16)
    loaded.load(str(path))
    assert loaded.ACTION_SCHEMA == "map_dqn_v1"


def reward_state() -> dict:
    """Reward screen with two claimable items and a proceed option."""
    return {
        "state_type": "rewards",
        "rewards": {
            "items": [{"index": 0, "type": "card"}, {"index": 1, "type": "gold"}],
            "can_proceed": True,
        },
        "player": {"hp": 40, "max_hp": 80, "gold": 99},
        "run": {"floor": 3, "act": 1},
    }


def reward_recordings() -> list[tuple[dict, dict]]:
    """Recorded reward claims in the mod's API action schema."""
    return [
        (reward_state(), {"type": "claim_reward", "index": 1}),
        (reward_state(), {"type": "claim_reward", "index": 0}),
    ]


def names_for(*argv: str) -> list[str]:
    """Resolve target names from a CLI selection (recordings arg is required)."""
    return resolve_target_names(parse_args(["--recordings", "x", *argv]))


def test_resolve_target_names_selection_cases():
    """The CLI selection maps to the documented target set."""
    assert names_for() == ["battle"]
    assert names_for("--screen", "map") == ["map"]
    assert names_for("--screen", "map", "--screen", "shop") == ["map", "shop"]
    assert names_for("--screen", "all") == ["battle", *SCREEN_NAMES]
    assert names_for("--screen", "all", "--screens-only") == list(SCREEN_NAMES)
    with pytest.raises(SystemExit):
        names_for("--screens-only")  # battle removed, nothing left


def test_bucket_samples_routes_each_state_to_its_target(tmp_path):
    """One pass over recordings routes each state to exactly one controlling target."""
    targets = [
        PretrainTarget("battle", None, Path("x"), is_battle_policy_state),
        PretrainTarget("map", None, Path("x"), lambda s: screen_name_for_state(s) == "map"),
        PretrainTarget("reward", None, Path("x"), lambda s: screen_name_for_state(s) == "reward"),
    ]
    battle_record = (battle_state(), {"type": "play_card", "card_index": 0, "target": "ENEMY_0"})
    path = tmp_path / "mixed.jsonl"
    write_jsonl(path, [battle_record, *map_recordings(), *reward_recordings()])

    bucket_samples([path], targets, limit=None)

    by_name = {target.name: target for target in targets}
    assert len(by_name["battle"].samples) == 1
    assert len(by_name["map"].samples) == 2
    assert len(by_name["reward"].samples) == 2


def test_main_trains_multiple_screen_agents(tmp_path, monkeypatch):
    """`--screen map --screen reward` pretrains both, each to its own checkpoint."""
    path = tmp_path / "mixed.jsonl"
    write_jsonl(path, [*map_recordings() * 4, *reward_recordings() * 4])
    monkeypatch.chdir(tmp_path)

    main(["--recordings", str(path), "--screen", "map", "--screen", "reward", "--epochs", "1", "--seed", "0"])

    assert (tmp_path / "checkpoints" / "mapAgent" / "PPO" / "mapagent_latest.pt").exists()
    assert (tmp_path / "checkpoints" / "rewardAgent" / "PPO" / "rewardagent_latest.pt").exists()
    assert not (tmp_path / "checkpoints" / "battleAgent").exists()


def test_main_screen_all_includes_battle_and_skips_empty(tmp_path, monkeypatch):
    """`--screen all` trains battle from battle recordings; empty screens are skipped."""
    path = tmp_path / "battle_only.jsonl"
    write_jsonl(path, sample_recordings() * 4)
    monkeypatch.chdir(tmp_path)

    main(["--recordings", str(path), "--screen", "all", "--battle-agent", "DQN", "--epochs", "1", "--seed", "0"])

    assert (tmp_path / "checkpoints" / "battleAgent" / "DQN" / "battleagent_latest.pt").exists()
    # Recordings held no screen states, so those targets produced no files.
    assert not (tmp_path / "checkpoints" / "mapAgent").exists()


def test_main_rejects_out_with_multiple_targets(tmp_path):
    """--out is ambiguous when more than one agent is trained."""
    path = tmp_path / "rec.jsonl"
    write_jsonl(path, map_recordings())
    with pytest.raises(SystemExit):
        main(["--recordings", str(path), "--screen", "all", "--out", str(tmp_path / "x.pt")])

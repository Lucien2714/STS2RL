"""Tests for evaluation seed generation."""

from sts2rl.evaluation.seeds import evaluation_client_episode_seeds


def test_client_episode_seeds_are_grouped_by_client():
    """Base seeds should be grouped into per-client episode lists."""
    assert evaluation_client_episode_seeds(3, 2, "S") == [
        ["S", "S_2", "S_3"],
        ["S_4", "S_5", "S_6"],
    ]


def test_random_client_episode_seed_count():
    """Random seeds should be unique across all client episodes."""
    seeds = evaluation_client_episode_seeds(4, 3, None)
    assert len(seeds) == 3
    assert all(len(client_seeds) == 4 for client_seeds in seeds)
    assert len({seed for client_seeds in seeds for seed in client_seeds}) == 12

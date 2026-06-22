"""Deterministic and random seed helpers for evaluation episodes."""

import secrets
import string

SEED_ALPHABET = string.ascii_uppercase + string.digits
SEED_LENGTH = 10


def evaluation_seed(base_seed: str | None, episode_index: int) -> str:
    """Return an episode seed derived from a base seed or random source."""
    if base_seed is None:
        return random_evaluation_seed()
    if episode_index == 1:
        return base_seed
    return f"{base_seed}_{episode_index}"


def evaluation_episode_seeds(episodes: int, base_seed: str | None) -> list[str]:
    """Return one seed per episode."""
    return [evaluation_seed(base_seed, episode_index) for episode_index in range(1, episodes + 1)]


def evaluation_client_episode_seeds(
    episodes: int,
    client_count: int,
    base_seed: str | None,
) -> list[list[str]]:
    """Group episode seeds by client for multi-client evaluation."""
    all_seeds = evaluation_episode_seeds(episodes * client_count, base_seed)
    return [all_seeds[index * episodes : (index + 1) * episodes] for index in range(client_count)]


def random_evaluation_seed() -> str:
    """Generate a random uppercase alphanumeric seed accepted by STS2."""
    return "".join(secrets.choice(SEED_ALPHABET) for _ in range(SEED_LENGTH))

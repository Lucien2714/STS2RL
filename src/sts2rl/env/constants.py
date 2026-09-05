"""Shared environment constants."""

COMBAT_STATE_TYPES = {"monster", "elite", "boss"}
BATTLE_STATE_TYPES = COMBAT_STATE_TYPES | {"hand_select"}
BATTLE_REWARD_STATE_TYPES = {"rewards", "card_reward"}

# Reward magnitudes.  These are kept within roughly one order of magnitude of
# each other: PPO shares one trunk between the policy and the value head, and
# an outcome term far larger than the per-step terms makes the value loss
# dominate the shared gradient and wash out the policy signal.  Scale the whole
# block together rather than any single entry.
BATTLE_WIN_REWARD = 20.0
BATTLE_LOSS_PENALTY = 15.0
ENEMY_KILL_REWARD = 1.0
ENEMY_DAMAGE_REWARD = 0.1
BATTLE_HP_LOSS_PENALTY = 0.1
BATTLE_GOLD_LOSS_PENALTY = 0.01
BATTLE_MAX_HP_LOSS_PENALTY = 0.5
POTION_USE_PENALTY = 0.5
UNSPENT_ENERGY_PENALTY = 0.5

FLOOR_PROGRESS_REWARD = 1.0
RUN_HP_CHANGE_REWARD = 0.02
GAME_OVER_PENALTY = 1.0

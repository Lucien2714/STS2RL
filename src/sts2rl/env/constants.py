"""Shared environment constants."""

COMBAT_STATE_TYPES = {"monster", "elite", "boss"}
BATTLE_STATE_TYPES = COMBAT_STATE_TYPES | {"hand_select"}
BATTLE_REWARD_STATE_TYPES = {"rewards", "card_reward"}

BATTLE_GOLD_LOSS_PENALTY = 0.1
BATTLE_MAX_HP_LOSS_PENALTY = 5.0
BATTLE_LOSS_PENALTY = 150.0

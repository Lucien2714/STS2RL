"""Shared environment constants."""

COMBAT_STATE_TYPES = {"monster", "elite", "boss"}
BATTLE_STATE_TYPES = COMBAT_STATE_TYPES | {"hand_select"}

# Reward magnitudes.
#
# The run is scored on the only thing it is trying to do: climb.  Everything
# here is expressed relative to one node, so the whole block scales together by
# changing NODE_PROGRESS_REWARD and keeping the ratios.
#
# A full act is roughly 16 nodes plus a boss, so an act is worth about 26 and a
# three-act win about 80.  Returns stay monotone in progress, which is what
# makes them easy for the critic to fit.
NODE_PROGRESS_REWARD = 1.0
BOSS_VICTORY_REWARD = 10.0

# Standing still otherwise costs nothing: a measured run spent 400 steps
# toggling one selection screen for exactly 0.000 reward.  One node is worth
# 100 steps of loitering, so this never outweighs real progress.
STEP_COST = 0.01

# HP is the resource the whole run spends.  Without this, clearing a node at
# 1 HP scores the same as clearing it untouched, and the critic has to learn
# the difference from the deaths it causes several nodes later.  Losing a
# third of a starting health bar costs about one node.  Set to 0.0 to score
# progress alone.
HP_CHANGE_REWARD = 0.03

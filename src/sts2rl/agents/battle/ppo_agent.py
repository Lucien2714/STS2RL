"""Backward-compatible re-export of the PPO candidate-action agent.

The PPO agent moved to :mod:`sts2rl.agents.candidate_ppo_agent` when the
candidate-action framework was generalized beyond battle (a bare
``PPOCandidateAgent()`` is still the battle agent). Importing the battle-named
classes from here still works.
"""

from sts2rl.agents.candidate_ppo_agent import (
    BattlePPOAgent,
    BattlePPOPolicy,
    CandidatePPOPolicy,
    PPOBattleAgent,
    PPOCandidateAgent,
    PPORolloutCollector,
    layer_init,
)

__all__ = [
    "BattlePPOAgent",
    "BattlePPOPolicy",
    "CandidatePPOPolicy",
    "PPOBattleAgent",
    "PPOCandidateAgent",
    "PPORolloutCollector",
    "layer_init",
]

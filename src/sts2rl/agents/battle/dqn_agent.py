"""Backward-compatible re-export of the DQN candidate-action agent.

The DQN agent moved to :mod:`sts2rl.agents.candidate_dqn_agent` when the
candidate-action framework was generalized beyond battle (a bare
``DQNCandidateAgent()`` is still the battle agent). Importing the battle-named
classes from here still works.
"""

from sts2rl.agents.candidate_dqn_agent import (
    BattleDQNAgent,
    BattleQNetwork,
    CandidateQNetwork,
    DQNBattleAgent,
    DQNCandidateAgent,
    SharedReplayCollector,
)

__all__ = [
    "BattleDQNAgent",
    "BattleQNetwork",
    "CandidateQNetwork",
    "DQNBattleAgent",
    "DQNCandidateAgent",
    "SharedReplayCollector",
]

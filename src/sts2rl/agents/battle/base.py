"""Backward-compatible re-export of the screen-agnostic candidate-action base.

``CandidateActionAgent`` moved to :mod:`sts2rl.agents.candidate_agent` when the
candidate-action framework was generalized beyond battle. Importing it from here
still works.
"""

from sts2rl.agents.candidate_agent import CandidateActionAgent

__all__ = ["CandidateActionAgent"]

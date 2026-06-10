"""Map agent implementations."""

from sts2rl.agents.map.rule_based import MapPolicy

RuleBasedMapAgent = MapPolicy

__all__ = ["MapPolicy", "RuleBasedMapAgent"]

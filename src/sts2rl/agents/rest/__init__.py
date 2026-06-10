"""Rest site agent implementations."""

from sts2rl.agents.rest.rule_based import RestPolicy

RuleBasedRestAgent = RestPolicy

__all__ = ["RestPolicy", "RuleBasedRestAgent"]

"""Event agent implementations."""

from sts2rl.agents.event.rule_based import EventPolicy

RuleBasedEventAgent = EventPolicy

__all__ = ["EventPolicy", "RuleBasedEventAgent"]

"""Fallback agent implementations."""

from sts2rl.agents.default.rule_based import DefaultPolicy

DefaultAgent = DefaultPolicy

__all__ = ["DefaultAgent", "DefaultPolicy"]

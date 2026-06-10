"""Shop agent implementations."""

from sts2rl.agents.shop.rule_based import ShopPolicy

RuleBasedShopAgent = ShopPolicy

__all__ = ["ShopPolicy", "RuleBasedShopAgent"]

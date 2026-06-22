"""Rule-based rest-site policy."""

import logging

logger = logging.getLogger(__name__)


class RestPolicy:
    """Rest when health is low, otherwise prefer upgrading when available."""

    def choose_action(self, state: dict) -> dict:
        """Choose a rest-site option from the enabled options."""
        rest_site = state.get("rest_site", {})
        options = rest_site.get("options", [])
        player = state.get("player", {})
        hp = player.get("hp", player.get("current_hp", 0))
        max_hp = player.get("max_hp", 1)

        hp_ratio = hp / max_hp
        logger.debug(
            "RestPolicy: choosing action options=%d hp=%s max_hp=%s hp_ratio=%.2f",
            len(options),
            hp,
            max_hp,
            hp_ratio,
        )

        if not options:
            action = {"type": "proceed"}
            logger.debug("RestPolicy: selected action=%s", action)
            return action

        enabled_options = [option for option in options if option.get("is_enabled", True)]

        if not enabled_options:
            action = {"type": "proceed"}
            logger.debug("RestPolicy: selected action=%s", action)
            return action

        if hp_ratio < 0.66:
            rest_option = self._find_option(enabled_options, {"rest"})
            if rest_option is not None:
                action = {
                    "type": "choose_rest_option",
                    "index": rest_option.get("index", 0),
                }
                logger.debug("RestPolicy: selected action=%s", action)
                return action

        upgrade_option = self._find_option(enabled_options, {"smith", "upgrade"})
        chosen_option = upgrade_option or enabled_options[0]
        action = {
            "type": "choose_rest_option",
            "index": chosen_option.get("index", 0),
        }
        logger.debug("RestPolicy: selected action=%s", action)
        return action

    def _find_option(self, options: list[dict], ids: set[str]) -> dict | None:
        """Find the first option whose id or name matches one of the desired ids."""
        for option in options:
            option_id = str(option.get("id", "")).lower()
            option_name = str(option.get("name", "")).lower()
            if option_id in ids or option_name in ids:
                return option

        return None

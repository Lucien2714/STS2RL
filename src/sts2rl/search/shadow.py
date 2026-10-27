"""A simulator that follows a real game client, so a fight on it can be searched.

The real game cannot branch, so ``CombatSearch`` cannot run against it.  The
simulator hosts the game's own rules engine and draws from the same RNG, so a
simulator run started with the same seed and sent the same actions shows the
same states: two recorded human runs replayed this way matched the real game
through floor 11 (47 combat actions) and floor 27 (233). ``ShadowEnv`` keeps
such a run beside a real one, and the search plays the real fight from the
simulator's copy of it.

**Nothing can put the copy back once it parts.**  The simulator has no way to
load a state, only to start a run from a seed, so a run whose copy diverged is
played by the actor from there on: a search over the wrong fight is worse than
no search.  Agreement is therefore checked at every decision, on what the real
game shows (``fingerprint``), and a fight is searched only from a decision the
two agree on.

**A disagreement is not yet a divergence.**  The two step through some screens
differently while drawing nothing: the simulator leaves a reward screen as soon
as its card is taken and confirms a full selection itself, where the real
client stops on both and waits for the ``proceed`` or confirm the agent then
sends.  Those actions reach the simulator too, where they are refused and change
nothing, and the next decision agrees again.  So a copy is given up only after
``max_disagreements`` consecutive disagreeing decisions -- or at once, when the
real run was joined rather than started, or the run asks for modifiers the
simulator cannot set.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sts2rl.actions import GameAction
from sts2rl.env.game_env import GameEnv
from sts2rl.env.mcp_client import STS2ClientError
from sts2rl.env.reset import ResetSpec
from sts2rl.env.types import EnvStep, RawState

LOGGER = logging.getLogger(__name__)

# Screens whose listed choices are worth comparing, and the list each keeps them in.
_CHOICE_LISTS = {
    "card_reward": "cards",
    "rewards": "items",
    "shop": "items",
    "event": "options",
    "rest_site": "options",
    "map": "next_options",
}


def fingerprint(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return the visible facts two copies of one run must agree on.

    Display text is left out: the real client and the simulator render it
    separately.  Everything a fight is decided on is in: hand, energy, piles,
    statuses, potions, and each enemy's HP, block, statuses and intents.
    """
    run = raw.get("run") or {}
    player = raw.get("player") or {}
    battle = raw.get("battle") or {}
    facts: dict[str, Any] = {
        "state_type": raw.get("state_type"),
        "act": run.get("act"),
        "floor": run.get("floor"),
        "hp": player.get("hp"),
        "max_hp": player.get("max_hp"),
        "block": player.get("block"),
        "gold": player.get("gold"),
        "energy": player.get("energy"),
        "hand": [_card(card) for card in player.get("hand") or []],
        "piles": [player.get(f"{pile}_pile_count") for pile in ("draw", "discard", "exhaust")],
        "status": _powers(player.get("status")),
        "potions": [potion.get("id") for potion in player.get("potions") or []],
        "relics": [relic.get("id") for relic in player.get("relics") or []],
        "round": battle.get("round"),
        "enemies": [
            (
                enemy.get("entity_id"),
                enemy.get("hp"),
                enemy.get("block"),
                _powers(enemy.get("status")),
                [(intent.get("type"), intent.get("label")) for intent in enemy.get("intents") or []],
            )
            for enemy in battle.get("enemies") or []
        ],
    }
    for screen, key in _CHOICE_LISTS.items():
        block = raw.get(screen)
        if isinstance(block, dict) and isinstance(block.get(key), list):
            facts[screen] = [_choice(item) for item in block[key] if isinstance(item, dict)]
    return facts


def _card(card: Mapping[str, Any]) -> tuple[Any, ...]:
    return (card.get("id"), card.get("current_upgrade_level"), card.get("cost"))


def _powers(powers: Any) -> list[tuple[Any, Any]]:
    return [(power.get("id"), power.get("amount")) for power in powers or [] if isinstance(power, dict)]


def _choice(item: Mapping[str, Any]) -> tuple[Any, ...]:
    identity = item.get("id") or item.get("card_id") or item.get("text_key") or item.get("type")
    return (identity, item.get("col"), item.get("row"))


@dataclass
class ShadowStats:
    """One episode of following: how far the copy kept up, and where it was given up."""

    agreed: int = 0
    disagreed: int = 0
    refused: int = 0
    lost_at: dict[str, Any] | None = None
    differences: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreed": self.agreed,
            "disagreed": self.disagreed,
            "refused": self.refused,
            "lost_at": self.lost_at,
            "differences": self.differences,
        }


class ShadowEnv:
    """A real ``GameEnv`` that also plays every accepted action on a simulator.

    It stands where ``EpisodeRunner`` expects a ``GameEnv``: every state the
    runner sees is the real game's.  ``sim`` is the simulator copy, and is what
    a ``SimulatorSearchEnv`` is built on; ``agrees`` is the gate the search
    asks before using it.
    """

    def __init__(self, real: GameEnv, sim: GameEnv, *, max_disagreements: int = 3) -> None:
        if sim.backend != "sim":
            raise ValueError("the shadow must be a simulator (backend 'sim')")
        if max_disagreements < 1:
            raise ValueError("max_disagreements must be positive")
        self.real = real
        self.sim = sim
        self.max_disagreements = max_disagreements
        self.alive = False
        self.stats = ShadowStats()
        self.episodes: list[dict[str, Any]] = []
        self._sim_state: RawState | None = None
        self._streak = 0
        self._started = False

    # --- the GameEnv surface EpisodeRunner uses -------------------------------

    @property
    def backend(self) -> str:
        return self.real.backend

    @property
    def client(self):
        return self.real.client

    @property
    def reused_active_run(self) -> bool:
        return self.real.reused_active_run

    def reset(self, spec: ResetSpec | None = None) -> RawState:
        spec = spec or ResetSpec()
        self.alive = False
        self._sim_state = None
        state = self.real.reset(spec)
        # Only a reset that started a run opens a record: a failed one is retried
        # by the caller and is not an episode.
        if self._started:
            self.episodes.append(self.stats.to_dict())
        self._started = True
        self.stats = ShadowStats()
        self._streak = 0
        if self.real.reused_active_run:
            self._lose("the real client joined a run in progress")
        elif spec.modifiers:
            self._lose(f"the simulator cannot set modifiers {list(spec.modifiers)}")
        elif spec.run_seed is None:
            self._lose("the run has no seed to start the simulator from")
        else:
            try:
                self._sim_state = self.sim.reset(
                    ResetSpec(character=spec.character, run_seed=spec.run_seed, ascension=spec.ascension)
                )
            except (STS2ClientError, ValueError) as error:
                self._lose(f"the simulator did not start the run: {error}")
            else:
                self.alive = True
        return state

    def step(self, action: GameAction) -> EnvStep:
        result = self.real.step(action)
        if self.alive and not result.info.get("action_error"):
            # The simulator is a helper: its failure costs the search, never the episode.
            try:
                mirrored = self.sim.step(action)
            except STS2ClientError as error:
                self._lose(f"the simulator failed: {error}")
                return result
            if mirrored.info.get("action_error"):
                # The screen-flow case: the simulator is already past the screen
                # the real client stopped on.  Its state did not move.
                self.stats.refused += 1
            self._sim_state = mirrored.raw_state
        return result

    def get_state(self) -> RawState:
        return self.real.get_state()

    def get_player_detail(self) -> RawState | None:
        return self.real.get_player_detail()

    def is_end_state(self) -> bool:
        return self.real.is_end_state()

    def close(self) -> None:
        self.real.close()
        self.sim.close()

    def __enter__(self) -> "ShadowEnv":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # --- following -------------------------------------------------------------

    def agrees(self, real_state: Mapping[str, Any]) -> bool:
        """Return whether the simulator shows what the real game shows, and count it.

        Called once per decision.  A run of ``max_disagreements`` disagreeing
        decisions gives the copy up for the rest of the episode.
        """
        if not self.alive or self._sim_state is None:
            return False
        real, copy = fingerprint(real_state), fingerprint(self._sim_state)
        if real == copy:
            self.stats.agreed += 1
            self._streak = 0
            return True
        self.stats.disagreed += 1
        self._streak += 1
        if self._streak >= self.max_disagreements:
            self.stats.differences = {
                key: {"real": real.get(key), "sim": copy.get(key)}
                for key in sorted(set(real) | set(copy))
                if real.get(key) != copy.get(key)
            }
            self._lose(
                f"{self._streak} decisions in a row disagreed",
                floor=(real_state.get("run") or {}).get("floor"),
                state_type=real_state.get("state_type"),
            )
        return False

    def finish(self) -> list[dict[str, Any]]:
        """Return every episode's following record, this one included."""
        return [*self.episodes, self.stats.to_dict()]

    def _lose(self, reason: str, **where: Any) -> None:
        self.alive = False
        self.stats.lost_at = {"reason": reason, "decision": self.stats.agreed + self.stats.disagreed, **where}
        LOGGER.warning("shadow simulator given up: %s %s", reason, where or "")

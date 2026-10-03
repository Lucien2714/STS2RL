"""How good a fight state is, for the combat search's leaves.

This is a search heuristic, not a training reward.  The reward still scores only
climbing; this only decides which of two lines a search prefers when neither has
finished the fight.  It reads the raw state the API returns, so it needs no network
and cannot drift from what the agent is shown.

A finished fight is scored exactly: a loss is 0 and a win is worth more the more HP
and potions it keeps, in that order (CombatSolver's final ordering: survival, then
victory, then HP, then potions).  A fight in progress is scored as the chance of
winning it, ``sigma(w . phi(s))``, times what winning from here would be worth.  The
features are built around who falls first -- survival margin against the incoming
attack, the enemies' remaining health, the growth each side has banked -- because a
shallow search sees only what a turn or two does, and these are what carry the turns
after.  The weights start by hand and are refitted against the outcomes the search's
own fights reach (``fit_weights``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

RawState = Mapping[str, Any]

COMBAT_STATE_TYPES = frozenset({"monster", "elite", "boss", "hand_select"})

WIN_BASE = 0.7
WIN_HP_WEIGHT = 0.25
WIN_POTION_WEIGHT = 0.05

# Powers that change how a fight goes, read by id as the API sends them.
PLAYER_GROWTH = ("STRENGTH_POWER", "DEXTERITY_POWER")
DEBUFFS = ("VULNERABLE_POWER", "WEAK_POWER", "FRAIL_POWER")
ENEMY_GROWTH = ("STRENGTH_POWER", "RITUAL_POWER")

FEATURES = (
    "bias",
    "hp_ratio",
    "survival_margin",
    "lethal_incoming",
    "enemy_hp_ratio",
    "enemy_block_ratio",
    "player_strength",
    "player_dexterity",
    "player_buffs",
    "enemy_growth",
    "enemy_vulnerable",
    "enemy_weak",
    "player_debuffs",
    "potions",
    "status_cards",
)

# Starting weights, by hand: the race (own health against the incoming hit, the
# enemies' health) dominates, banked growth and debuffs move it a little.
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "bias": 0.0,
    "hp_ratio": 2.0,
    "survival_margin": 1.0,
    "lethal_incoming": -3.0,
    "enemy_hp_ratio": -4.0,
    "enemy_block_ratio": -1.0,
    "player_strength": 0.5,
    "player_dexterity": 0.3,
    "player_buffs": 0.3,
    "enemy_growth": -0.4,
    "enemy_vulnerable": 0.3,
    "enemy_weak": 0.3,
    "player_debuffs": -0.3,
    "potions": 0.2,
    "status_cards": -0.2,
}


def is_fight_over(state: RawState) -> bool:
    """Whether the fight has ended, in victory or death.

    The state type alone cannot say: a card that picks from a pile opens a
    ``card_select`` overlay mid-fight, reported without the battle block. The
    player block carries energy and the combat piles exactly while a fight is in
    progress, in the simulator as in the STS2MCP mod, so that is the test.
    """
    if state.get("state_type") == "game_over":
        return True
    return state.get("state_type") not in COMBAT_STATE_TYPES and "energy" not in _player(state)


def is_loss(state: RawState) -> bool:
    if state.get("state_type") == "game_over":
        block = state.get("game_over")
        victory = isinstance(block, Mapping) and block.get("victory") is True
        return not victory and state.get("terminal_reason") != "victory"
    return _number(_player(state).get("hp")) <= 0


def intent_damage(intent: Mapping[str, Any]) -> float:
    """The damage an intent will deal: its label is "11" for one hit, "7x2" for several."""
    if str(intent.get("type", "")).lower() not in ("attack", "deathblow"):
        return 0.0
    label = str(intent.get("label") or "").lower()
    per_hit, separator, hits = label.partition("x")
    try:
        return float(per_hit) * (float(hits) if separator else 1.0)
    except ValueError:
        return 0.0


def features(state: RawState) -> dict[str, float]:
    """The features of a fight in progress, each roughly on a unit scale."""
    player = _player(state)
    enemies = [
        enemy
        for enemy in _records((state.get("battle") or {}).get("enemies"))
        if _number(enemy.get("hp")) > 0
    ]
    hp = _number(player.get("hp"))
    max_hp = max(_number(player.get("max_hp")), 1.0)
    effective = hp + _number(player.get("block"))
    incoming = sum(
        intent_damage(intent) for enemy in enemies for intent in _records(enemy.get("intents"))
    )
    enemy_hp = sum(_number(e.get("hp")) for e in enemies)
    enemy_max = max(sum(_number(e.get("max_hp")) for e in enemies), 1.0)
    enemy_block = sum(_number(e.get("block")) for e in enemies)
    own = _powers(player)
    theirs = [_powers(e) for e in enemies]
    piles = [
        *_records(player.get("hand")),
        *_records(player.get("draw_pile")),
        *_records(player.get("discard_pile")),
    ]
    statuses = sum(
        _number(card.get("quantity")) or 1.0
        for card in piles
        if str(card.get("type", "")).lower() in ("status", "curse")
    )
    potions = [p for p in _records(player.get("potions")) if p.get("id")]
    return {
        "bias": 1.0,
        "hp_ratio": hp / max_hp,
        "survival_margin": max(-3.0, min(3.0, math.log1p(effective) - math.log1p(incoming))),
        "lethal_incoming": 1.0 if incoming >= effective > 0 or effective <= 0 else 0.0,
        "enemy_hp_ratio": enemy_hp / enemy_max,
        "enemy_block_ratio": enemy_block / enemy_max,
        "player_strength": own.get("STRENGTH_POWER", 0.0) / 5.0,
        "player_dexterity": own.get("DEXTERITY_POWER", 0.0) / 5.0,
        "player_buffs": sum(1 for name, kind in _power_kinds(player) if kind == "buff") / 5.0,
        "enemy_growth": sum(p.get(name, 0.0) for p in theirs for name in ENEMY_GROWTH) / 5.0,
        "enemy_vulnerable": min(sum(p.get("VULNERABLE_POWER", 0.0) for p in theirs), 3.0) / 3.0,
        "enemy_weak": min(sum(p.get("WEAK_POWER", 0.0) for p in theirs), 3.0) / 3.0,
        "player_debuffs": sum(min(own.get(name, 0.0), 3.0) for name in DEBUFFS) / 3.0,
        "potions": len(potions) / 3.0,
        "status_cards": statuses / 5.0,
    }


@dataclass(frozen=True)
class LeafEvaluator:
    """Scores a fight state in [0, 1]: exactly when it is over, by estimate when it is not."""

    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __call__(self, state: RawState) -> float:
        if is_fight_over(state):
            return self.terminal(state)
        return self.win_probability(state) * self._win_value(state)

    def terminal(self, state: RawState) -> float:
        if is_loss(state):
            return 0.0
        return self._win_value(state)

    def win_probability(self, state: RawState) -> float:
        phi = features(state)
        score = sum(self.weights.get(name, 0.0) * value for name, value in phi.items())
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))

    @staticmethod
    def _win_value(state: RawState) -> float:
        player = _player(state)
        hp_ratio = _number(player.get("hp")) / max(_number(player.get("max_hp")), 1.0)
        slots = max(_number(player.get("max_potion_slots")), 1.0)
        potions = len([p for p in _records(player.get("potions")) if p.get("id")]) / slots
        return WIN_BASE + WIN_HP_WEIGHT * max(0.0, hp_ratio) + WIN_POTION_WEIGHT * min(potions, 1.0)


def fit_weights(
    rows: Sequence[Mapping[str, float]],
    outcomes: Sequence[bool],
    *,
    l2: float = 1e-2,
    iterations: int = 500,
    learning_rate: float = 0.5,
) -> dict[str, float]:
    """Logistic regression of fight outcomes on features, by plain gradient descent.

    Few rows and fifteen features need nothing heavier, and keeping it dependency-free
    keeps the evaluator importable anywhere the search runs.
    """
    weights = {name: 0.0 for name in FEATURES}
    n = len(rows)
    if n == 0:
        return dict(DEFAULT_WEIGHTS)
    for _ in range(iterations):
        gradient = {name: 0.0 for name in FEATURES}
        for phi, won in zip(rows, outcomes):
            score = sum(weights[name] * phi.get(name, 0.0) for name in FEATURES)
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
            error = p - (1.0 if won else 0.0)
            for name in FEATURES:
                gradient[name] += error * phi.get(name, 0.0)
        for name in FEATURES:
            penalty = 0.0 if name == "bias" else l2 * weights[name]
            weights[name] -= learning_rate * (gradient[name] / n + penalty)
    return weights


def _player(state: RawState) -> Mapping[str, Any]:
    player = state.get("player")
    return player if isinstance(player, Mapping) else {}


def _powers(holder: Mapping[str, Any]) -> dict[str, float]:
    amounts: dict[str, float] = {}
    for power in _records(holder.get("status")):
        name = str(power.get("id", ""))
        amounts[name] = amounts.get(name, 0.0) + _number(power.get("amount"))
    return amounts


def _power_kinds(holder: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(p.get("id", "")), str(p.get("type", "")).lower())
        for p in _records(holder.get("status"))
    ]


def _records(value: object) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _number(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

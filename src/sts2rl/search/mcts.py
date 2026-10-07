"""Information-set MCTS over one fight, from a simulator branch point.

Each simulation restores the branch point, redraws everything the fight has not
revealed (``reseed``), and plays down the tree.  Because the hidden future is
resampled every time, a node's statistics average over the futures a player could
face instead of planning against the one the simulator happens to hold.

Three consequences shape the tree:

* **Nodes are keyed by what an action is, not where it sits.**  After a draw the hand
  differs between worlds, so "the card at index 2" names different cards; "Strike+1 at
  the Matriarch" names the same decision in every world.  Two copies of one card are
  one decision, which also removes their transpositions.
* **Availability, not parent visits, scales exploration.**  An action can be offered
  in some worlds and not others (it is in the hand only when drawn), so its UCB term
  counts the simulations in which it was available (Cowling et al., 2012).
* **Progressive widening** lets a node try a new action only once its visits allow it
  (``ceil(c * n ** alpha)`` children), so the second turn, where every world deals a
  different hand, is not spread across every card at once.

Exploration is weighed against values normalised to the range the tree has seen
(as MuZero does).  Leaf values live in [0, 1] but siblings typically differ by a few
hundredths, so a raw UCB lets the exploration term swamp them and spreads a small
budget almost evenly; normalised, the same budget concentrates.

The tree reaches ``turn_depth`` turns: a simulation stops after that many ``end_turn``
actions (the enemy turn resolved, the next hand dealt) or when the fight ends, and the
leaf is scored by the evaluator.  Unvisited actions are tried in a fixed preference
order, cards before ending the turn and potions after, so a fresh node plays its hand
out before it gives up its energy.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from sts2rl.actions import GameAction
from sts2rl.agents.action_space import LegalActionProvider, NoLegalActionsError
from sts2rl.agents.base import without_excluded
from sts2rl.search.evaluate import LeafEvaluator, is_fight_over

RawState = Mapping[str, Any]
ActionKey = tuple[Any, ...]


class SearchEnv(Protocol):
    """What the search needs from an environment: branch points, reseeds, and steps."""

    def snapshot(self) -> int: ...

    def restore(self, point: int, seed: int | None = None) -> RawState:
        """Return to ``point``, then ``reseed(seed)`` when a seed is given; the state after both."""
        ...

    def release(self, point: int) -> None: ...

    def reseed(self, seed: int) -> None: ...

    def step(self, action: GameAction) -> RawState: ...


@dataclass(frozen=True)
class MctsConfig:
    simulations: int = 200
    turn_depth: int = 2
    exploration: float = 0.7
    widening_c: float = 1.5
    widening_alpha: float = 0.5
    seed_pool: int = 32
    max_steps_per_simulation: int = 120
    # A wall-clock limit per decision, so trees of different depth can be compared at
    # equal cost: a deeper simulation takes longer. ``simulations`` stays the cap.
    seconds: float | None = None
    # False plans against the world the simulator holds: every simulation sees the
    # real draws and enemy rolls. That is cheating, and it is the point -- a fight a
    # clairvoyant search loses is one no amount of better play would win, so it
    # bounds what a fair search can reach.
    reseed: bool = True

    def __post_init__(self) -> None:
        if self.simulations < 1 or self.turn_depth < 1 or self.seed_pool < 1:
            raise ValueError("simulations, turn_depth and seed_pool must be positive")
        if self.widening_c <= 0 or not 0 < self.widening_alpha <= 1:
            raise ValueError("widening_c must be positive and widening_alpha in (0, 1]")


@dataclass
class Node:
    visits: int = 0
    value_sum: float = 0.0
    children: dict[ActionKey, Node] = field(default_factory=dict)
    # Simulations in which each action was offered here, expanded or not.
    available: dict[ActionKey, int] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    """The chosen root action and how the search spread over the root's candidates."""

    action: GameAction
    candidates: tuple[GameAction, ...]
    visits: tuple[int, ...]
    values: tuple[float, ...]
    root: Node

    @property
    def distribution(self) -> tuple[float, ...]:
        """Visit shares over the candidates; copies of one decision split its share."""
        total = sum(self.visits)
        return tuple(v / total for v in self.visits) if total else ()


def action_key(state: RawState, action: GameAction) -> ActionKey:
    """What an action is, independent of the index it happens to have in this world."""
    params = action.params
    player = state.get("player") if isinstance(state.get("player"), Mapping) else {}
    if action.action_type == "play_card":
        return ("play", _card_identity(player.get("hand"), params.get("card_index")), params.get("target"))
    if action.action_type in ("use_potion", "discard_potion"):
        return (action.action_type, _potion_identity(player.get("potions"), params.get("slot")), params.get("target"))
    if action.action_type == "combat_select_card":
        prompt = state.get("hand_select") if isinstance(state.get("hand_select"), Mapping) else {}
        return ("select", _card_identity(prompt.get("cards"), params.get("card_index")))
    if action.action_type == "select_card":
        prompt = state.get("card_select") if isinstance(state.get("card_select"), Mapping) else {}
        return ("pick", _card_identity(prompt.get("cards"), params.get("index")))
    return (action.action_type, *sorted(params.items()))


def preference(action: GameAction) -> int:
    """Order in which a node tries actions it has not tried: higher first."""
    return {
        "play_card": 3,
        "combat_select_card": 3,
        "select_card": 3,
        "combat_confirm_selection": 2,
        "confirm_selection": 2,
        "end_turn": 1,
        "use_potion": 1,
        "discard_potion": 0,
    }.get(action.action_type, 2)


class CombatSearch:
    """ISMCTS for one decision, run against an environment that can branch and reseed."""

    def __init__(
        self,
        config: MctsConfig | None = None,
        evaluator: Callable[[RawState, RawState, RawState], float] | None = None,
        candidates: Callable[[RawState], Sequence[GameAction]] | None = None,
        key: Callable[[RawState, GameAction], ActionKey] = action_key,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or MctsConfig()
        self.evaluate = evaluator or LeafEvaluator()
        self.candidates = candidates or LegalActionProvider().require_candidates
        self.key = key
        self.rng = rng or random.Random()
        self._low, self._high = math.inf, -math.inf

    def search(
        self,
        env: SearchEnv,
        state: RawState,
        simulations: int | None = None,
        exclude: Sequence[GameAction] = (),
    ) -> SearchResult:
        """Search one decision; ``simulations`` overrides the configured budget.

        ``exclude`` names root actions the game refused on this unchanged screen
        (``EpisodeRunner``).  They leave the root's candidates, and the
        simulations do not spend budget on them: the simulator may accept what
        the game refused, which is how the search chose the refused action in
        the first place.  The filter is by full parameters and comes before the
        grouping by key: two copies of one card are one decision, and the copy
        that was not refused stands for it -- it is the action the simulations
        play, so the key's visits and subtree are its own.  Below the root
        nothing changes.
        """
        budget = self.config.simulations if simulations is None else simulations
        if budget < 1:
            raise ValueError("simulations must be positive")
        candidates = tuple(self.candidates(state))
        if exclude:
            candidates = without_excluded(candidates, exclude)
        point = env.snapshot()
        # A fresh pool per decision: siblings are compared in the same worlds, and no
        # decision is planned against another's.
        seeds = [self.rng.getrandbits(32) for _ in range(self.config.seed_pool)]
        root = Node()
        self._low, self._high = math.inf, -math.inf
        deadline = None if self.config.seconds is None else time.perf_counter() + self.config.seconds
        try:
            for index in range(budget):
                if deadline is not None and index > 0 and time.perf_counter() > deadline:
                    break
                # Restore and reseed travel as one request against the simulator.
                seed = seeds[index % len(seeds)] if self.config.reseed else None
                world = env.restore(point, seed)
                self._simulate(env, root, world, tuple(exclude))
        finally:
            # The environment is left where the decision was taken, with its own hidden
            # state: the last simulation's world is not the game's.
            try:
                env.restore(point)
            finally:
                env.release(point)
        return self._result(root, state, candidates)

    def _simulate(
        self,
        env: SearchEnv,
        root: Node,
        state: RawState,
        root_exclude: tuple[GameAction, ...] = (),
    ) -> None:
        root_state = state
        path = [root]
        node = root
        turns = 0
        last_combat = state
        for _ in range(self.config.max_steps_per_simulation):
            if is_fight_over(state) or turns >= self.config.turn_depth:
                break
            last_combat = state
            try:
                offered = self.candidates(state)
                if node is root and root_exclude:
                    # Before grouping: the key's representative must be an
                    # action that was not refused.  A world where nothing is
                    # left ends the simulation here, never plays a refused one.
                    offered = without_excluded(offered, root_exclude)
            except NoLegalActionsError:
                break
            by_key: dict[ActionKey, GameAction] = {}
            for action in offered:
                by_key.setdefault(self.key(state, action), action)
            for key in by_key:
                node.available[key] = node.available.get(key, 0) + 1
            key = self._choose(node, by_key)
            action = by_key[key]
            state = env.step(action)
            if action.action_type == "end_turn":
                turns += 1
            node = node.children.setdefault(key, Node())
            path.append(node)
        value = self.evaluate(state, last_combat, root_state)
        self._low, self._high = min(self._low, value), max(self._high, value)
        for visited in path:
            visited.visits += 1
            visited.value_sum += value

    def _choose(self, node: Node, by_key: Mapping[ActionKey, GameAction]) -> ActionKey:
        if len(by_key) == 1:
            return next(iter(by_key))
        expanded = [key for key in by_key if key in node.children]
        untried = [key for key in by_key if key not in node.children]
        allowed = math.ceil(self.config.widening_c * (node.visits + 1) ** self.config.widening_alpha)
        if untried and (len(expanded) < allowed or not expanded):
            best = max(preference(by_key[key]) for key in untried)
            return self.rng.choice([key for key in untried if preference(by_key[key]) == best])
        return max(expanded, key=lambda key: self._ucb(node, key))

    def _ucb(self, node: Node, key: ActionKey) -> float:
        child = node.children[key]
        if child.visits == 0:
            return math.inf
        available = max(node.available.get(key, 1), 1)
        spread = self._high - self._low
        q = (child.value - self._low) / spread if spread > 1e-9 else 0.5
        return q + self.config.exploration * math.sqrt(math.log(available) / child.visits)

    def _result(self, root: Node, state: RawState, candidates: Sequence[GameAction]) -> SearchResult:
        keys = [self.key(state, action) for action in candidates]
        copies: dict[ActionKey, int] = {}
        for key in keys:
            copies[key] = copies.get(key, 0) + 1
        visits = []
        values = []
        for key in keys:
            child = root.children.get(key)
            visits.append((child.visits if child else 0) / copies[key])
            values.append(child.value if child else 0.0)
        best = max(
            range(len(candidates)),
            key=lambda i: (visits[i], values[i]),
        )
        return SearchResult(
            action=candidates[best],
            candidates=tuple(candidates),
            visits=tuple(visits),
            values=tuple(values),
            root=root,
        )


def _card_identity(cards: object, index: object) -> tuple[Any, ...] | None:
    for card in cards if isinstance(cards, list) else []:
        if isinstance(card, Mapping) and card.get("index") == index:
            return (
                card.get("id"),
                card.get("current_upgrade_level", 0),
                (card.get("enchantment") or {}).get("id") if isinstance(card.get("enchantment"), Mapping) else card.get("enchantment"),
            )
    return ("index", index)


def _potion_identity(potions: object, slot: object) -> Any:
    for potion in potions if isinstance(potions, list) else []:
        if isinstance(potion, Mapping) and potion.get("slot") == slot:
            return potion.get("id")
    return ("slot", slot)

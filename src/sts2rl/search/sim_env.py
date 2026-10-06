"""The simulator as a search environment, and a fight played by search."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sts2rl.actions import GameAction
from sts2rl.agents.action_space import NoLegalActionsError
from sts2rl.env.game_env import GameEnv
from sts2rl.env.state import extract_raw_state
from sts2rl.search.evaluate import is_fight_over, is_loss
from sts2rl.search.mcts import CombatSearch, Node, SearchResult

RawState = Mapping[str, Any]

BRANCHABLE = frozenset({"monster", "elite", "boss"})


class SimulatorSearchEnv:
    """Branch points, reseeds and steps against an STS2Simulator through ``GameEnv``.

    Steps go through ``GameEnv.step`` so a search plays exactly what the agent would
    play, including the screens the environment finishes on its own.
    """

    def __init__(self, env: GameEnv) -> None:
        self.env = env

    def snapshot(self) -> int:
        return self.env.client.sim_snapshot()

    def restore(self, point: int, seed: int | None = None) -> RawState:
        """Return to ``point``; with ``seed``, also redraw its hidden future, in one request.

        The state returned is the one after the reseed.  ``STS2Client.sim_restore``
        falls back to a separate reseed when the simulator does not confirm it.
        """
        return extract_raw_state(self.env.client.sim_restore(point, reseed=seed))

    def release(self, point: int) -> None:
        self.env.client.sim_release(point)

    def reseed(self, seed: int) -> None:
        self.env.client.sim_reseed(seed)

    def step(self, action: GameAction) -> RawState:
        return self.env.step(action).raw_state


@dataclass
class SearchedDecision:
    """One decision the search made, kept for distillation later."""

    state: RawState
    result: SearchResult
    seconds: float


@dataclass
class FightResult:
    won: bool
    final_state: RawState
    steps: int
    # The last state still in the fight: a lost fight ends on game_over, which no
    # longer shows the enemies.
    last_combat_state: RawState
    decisions: list[SearchedDecision] = field(default_factory=list)
    seconds: float = 0.0


def play_fight(
    env: GameEnv,
    search: CombatSearch,
    state: RawState,
    *,
    max_steps: int = 600,
    keep_decisions: bool = False,
) -> FightResult:
    """Play the fight in progress to its end, searching every decision that has a choice.

    A hand or pile pick (``hand_select``, a ``card_select`` mid-fight) blocks the
    game thread inside the decision, where no snapshot can be taken. The previous
    search already played through it in its simulations, so the pick is answered
    from that search's subtree: the most visited of the choices offered now.
    Without one, the first choice is taken.
    """
    sim = SimulatorSearchEnv(env)
    started = time.perf_counter()
    decisions: list[SearchedDecision] = []
    node: Node | None = None
    steps = 0
    last_combat = state
    while steps < max_steps and not is_fight_over(state):
        last_combat = state
        try:
            candidates = search.candidates(state)
        except NoLegalActionsError:
            break
        # Only a combat turn can be branched from: every other screen mid-fight (a hand or
        # pile pick) blocks the game thread inside the decision.
        if state.get("state_type") not in BRANCHABLE or len(candidates) == 1:
            action = _from_subtree(search, node, state, candidates)
        else:
            clock = time.perf_counter()
            result = search.search(sim, state)
            if keep_decisions:
                decisions.append(SearchedDecision(state, result, time.perf_counter() - clock))
            action = result.action
            node = result.root
        if node is not None:
            node = node.children.get(search.key(state, action))
        state = env.step(action).raw_state
        steps += 1
    return FightResult(
        won=is_fight_over(state) and not is_loss(state),
        final_state=state,
        steps=steps,
        last_combat_state=last_combat,
        decisions=decisions,
        seconds=time.perf_counter() - started,
    )


def _from_subtree(search: CombatSearch, node: Node | None, state: RawState, candidates) -> GameAction:
    if node is None or len(candidates) == 1:
        return candidates[0]
    scored = [
        (node.children[key].visits, index)
        for index, action in enumerate(candidates)
        if (key := search.key(state, action)) in node.children
    ]
    return candidates[max(scored)[1]] if scored else candidates[0]

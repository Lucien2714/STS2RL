"""Fights played by search inside a PPO run: the layered agent of step 2.

``SearchCombatAgent`` wraps one lane of the PPO agent. In the fights it is told to
search (every fight by default; ``rooms`` narrows it to elites and bosses) it chooses every action with the combat
search on its own lane's simulator and hands the choice to PPO as an external
action: the critic learns from those steps, the policy does not. Everywhere else
-- the map, events, shops, rewards, and any fight left out of ``rooms`` -- PPO
chooses as before.

The point is the critic. A fight lost to a misplayed hand teaches the critic that
the deck or the path before it was worse than it was; with the fight played
well, what reaches the map choice is closer to what the choice was worth.

Every searched decision can be recorded with the root's visit distribution, in
the cleaned-decision line format plus ``target_distribution``, for distillation.
"""

from __future__ import annotations

import gzip
import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from sts2rl.actions import GameAction
from sts2rl.agents.base import Agent, Transition
from sts2rl.env.game_env import GameEnv
from sts2rl.env.types import GameObservation
from sts2rl.search.evaluate import is_fight_over
from sts2rl.search.mcts import CombatSearch, Node, SearchResult
from sts2rl.search.sim_env import BRANCHABLE, SimulatorSearchEnv, _from_subtree

SEARCHED_ROOMS = frozenset({"monster", "elite", "boss"})


class ExternalLane(Protocol):
    """What the wrapper needs from a PPO lane: the Agent calls plus external actions."""

    def reset(self, initial_state: GameObservation) -> None: ...

    def choose_action(self, state: GameObservation) -> GameAction: ...

    def choose_external(self, state: GameObservation, action: GameAction) -> GameAction: ...

    def observe(self, transition: Transition) -> None: ...

    def discard_decision(self) -> None: ...

    def finish_episode(self, final_state: GameObservation, truncated: bool) -> None: ...


class SearchDecisionRecorder:
    """Appends searched decisions to a ``decisions.jsonl.gz`` shared by every lane.

    Each line is a cleaned decision (``offline.clean.Decision.to_json``) whose
    ``expert_index`` is the action the search played, plus ``target_distribution``:
    the root's visit shares over the same candidates, the soft label distillation
    trains on. One gzip member per line, so lanes append without coordinating
    beyond a lock, and a killed run loses at most the line being written.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.written = 0

    def record(
        self,
        observation: GameObservation,
        result: SearchResult,
        *,
        run_id: str,
        step_index: int,
    ) -> None:
        state = observation.raw_state
        run = state.get("run") if isinstance(state.get("run"), Mapping) else {}
        chosen = result.action.to_dict()
        line = {
            "run_id": run_id,
            "step_index": step_index,
            "state_type": state.get("state_type"),
            "act": run.get("act"),
            "floor": run.get("floor"),
            "raw_state": dict(state),
            "player_detail": None if observation.player_detail is None else dict(observation.player_detail),
            "candidates": [candidate.to_dict() for candidate in result.candidates],
            "expert_index": next(
                i for i, candidate in enumerate(result.candidates) if candidate.to_dict() == chosen
            ),
            "next_step_index": None,
            "steps_to_run_end": 0,
            "is_run_final_decision": False,
            "target_distribution": list(result.distribution),
            "search_values": list(result.values),
        }
        text = json.dumps(line) + "\n"
        with self._lock:
            with gzip.open(self.path, "at", encoding="utf-8") as handle:
                handle.write(text)
            self.written += 1


class SearchCombatAgent(Agent):
    """One lane: search plays the chosen fights, PPO plays everything else."""

    def __init__(
        self,
        lane: ExternalLane,
        env: GameEnv,
        search: CombatSearch,
        *,
        rooms: frozenset[str] = SEARCHED_ROOMS,
        recorder: SearchDecisionRecorder | None = None,
        run_label: str = "lane",
    ) -> None:
        self.lane = lane
        self.sim = SimulatorSearchEnv(env)
        self.search = search
        self.rooms = rooms
        self.recorder = recorder
        self.run_label = run_label
        self.searched = 0
        self._episode = 0
        self._step = 0
        self._in_fight = False
        self._node: Node | None = None
        self._node_before: Node | None = None

    def reset(self, initial_state: GameObservation) -> None:
        self._episode += 1
        self._step = 0
        self._in_fight = False
        self._node = self._node_before = None
        self.lane.reset(initial_state)

    def choose_action(self, state: GameObservation) -> GameAction:
        raw = state.raw_state
        state_type = raw.get("state_type")
        self._step += 1
        if state_type in BRANCHABLE:
            # A fight's own screen says which room it is; the overlays inside it
            # (a hand or pile pick) do not, so the room is remembered until it ends.
            self._in_fight = state_type in self.rooms
        elif is_fight_over(raw):
            self._in_fight = False
        if not self._in_fight:
            self._node = self._node_before = None
            return self.lane.choose_action(state)

        candidates = self.search.candidates(raw)
        self._node_before = self._node
        if state_type in BRANCHABLE and len(candidates) > 1:
            result = self.search.search(self.sim, raw)
            self.searched += 1
            if self.recorder is not None:
                self.recorder.record(
                    state, result, run_id=f"{self.run_label}-{self._episode}", step_index=self._step
                )
            action = result.action
            self._node = result.root
        else:
            # Nothing to branch from (a single candidate, or an overlay that blocks
            # the game thread): the last search already played through it.
            action = _from_subtree(self.search, self._node, raw, candidates)
        if self._node is not None:
            self._node = self._node.children.get(self.search.key(raw, action))
        return self.lane.choose_external(state, action)

    def observe(self, transition: Transition) -> None:
        self.lane.observe(transition)

    def discard_decision(self) -> None:
        # The action was refused and the screen did not move: the tree did not move
        # either, so the retry must answer from where it stood.
        self._node = self._node_before
        self.lane.discard_decision()

    def finish_episode(self, final_state: GameObservation, truncated: bool) -> None:
        self._in_fight = False
        self._node = self._node_before = None
        self.lane.finish_episode(final_state, truncated)

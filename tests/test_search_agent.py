"""The layered agent: search plays the chosen fights, PPO plays everything else."""

from __future__ import annotations

import gzip
import json

from sts2rl.actions import GameAction
from sts2rl.agents.base import Transition
from sts2rl.env.types import GameObservation
from sts2rl.search import CombatSearch, MctsConfig, Node, SearchCombatAgent, SearchDecisionRecorder, SearchResult

STRIKE, DEFEND = GameAction("play_card", card_index=0), GameAction("play_card", card_index=1)
END = GameAction("end_turn")
PICK_A, PICK_B = GameAction("combat_select_card", card_index=0), GameAction("combat_select_card", card_index=1)
NODE = GameAction("choose_map_node", index=0)


def fight(room: str) -> dict:
    return {"state_type": room, "player": {"energy": 3, "hand": []}, "battle": {"is_play_phase": True}}


HAND_SELECT = {"state_type": "hand_select", "player": {"energy": 1}, "hand_select": {"cards": []}}
MAP = {"state_type": "map", "player": {"hp": 50}}


def candidates(state: dict) -> list[GameAction]:
    if state["state_type"] == "hand_select":
        return [PICK_A, PICK_B]
    if state["state_type"] == "map":
        return [NODE]
    return [STRIKE, DEFEND, END]


def key(state: dict, action: GameAction) -> tuple:
    return (action.action_type, *sorted(action.params.items()))


class FakeSearch(CombatSearch):
    """Plays DEFEND, and its tree says the hand pick after DEFEND is PICK_B."""

    def __init__(self) -> None:
        super().__init__(MctsConfig(simulations=1), candidates=candidates, key=key)
        self.calls = 0

    def search(self, env, state):
        self.calls += 1
        offered = tuple(candidates(state))
        after_defend = Node(visits=10, children={key(HAND_SELECT, PICK_A): Node(visits=2),
                                                  key(HAND_SELECT, PICK_B): Node(visits=8)})
        root = Node(visits=10, children={key(state, DEFEND): after_defend})
        return SearchResult(action=DEFEND, candidates=offered, visits=(1, 8, 1), values=(0.1, 0.5, 0.2), root=root)


class FakeLane:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def reset(self, initial_state):
        self.calls.append(("reset", None))

    def choose_action(self, state):
        self.calls.append(("ppo", state.raw_state["state_type"]))
        return candidates(state.raw_state)[0]

    def choose_external(self, state, action):
        self.calls.append(("external", action.to_dict()))
        return action

    def observe(self, transition):
        self.calls.append(("observe", None))

    def discard_decision(self):
        self.calls.append(("discard", None))

    def finish_episode(self, final_state, truncated):
        self.calls.append(("finish", truncated))


def agent(tmp_path=None, rooms=frozenset({"elite", "boss"})):
    lane, search = FakeLane(), FakeSearch()
    recorder = SearchDecisionRecorder(tmp_path / "decisions.jsonl.gz") if tmp_path else None
    wrapped = SearchCombatAgent(lane, env=object(), search=search, rooms=rooms, recorder=recorder)
    wrapped.reset(GameObservation(MAP))
    return wrapped, lane, search


def obs(state: dict) -> GameObservation:
    return GameObservation(state)


def test_every_fight_is_searched_by_default():
    lane, search = FakeLane(), FakeSearch()
    wrapped = SearchCombatAgent(lane, env=object(), search=search)
    wrapped.reset(obs(MAP))
    for room in ("monster", "elite", "boss"):
        wrapped.choose_action(obs(fight(room)))
    assert search.calls == 3


def test_outside_searched_fights_ppo_chooses_and_nothing_is_searched():
    wrapped, lane, search = agent()
    wrapped.choose_action(obs(MAP))
    wrapped.choose_action(obs(fight("monster")))
    wrapped.choose_action(obs(HAND_SELECT))  # an overlay inside an ordinary fight
    assert search.calls == 0
    assert [call for call in lane.calls if call[0] != "reset"] == [
        ("ppo", "map"), ("ppo", "monster"), ("ppo", "hand_select")
    ]


def test_an_elite_fight_is_searched_and_handed_to_ppo_as_external():
    wrapped, lane, search = agent()
    action = wrapped.choose_action(obs(fight("elite")))
    assert search.calls == 1
    assert action.to_dict() == DEFEND.to_dict()
    assert lane.calls[-1] == ("external", DEFEND.to_dict())


def test_an_overlay_inside_a_searched_fight_is_answered_from_the_tree():
    wrapped, lane, search = agent()
    wrapped.choose_action(obs(fight("boss")))
    action = wrapped.choose_action(obs(HAND_SELECT))
    assert search.calls == 1  # no search on a screen that blocks the game thread
    assert action.to_dict() == PICK_B.to_dict()
    assert lane.calls[-1] == ("external", PICK_B.to_dict())


def test_a_refused_action_rewinds_the_tree_for_the_retry():
    wrapped, lane, search = agent()
    wrapped.choose_action(obs(fight("boss")))
    wrapped.choose_action(obs(HAND_SELECT))
    wrapped.discard_decision()
    assert wrapped.choose_action(obs(HAND_SELECT)).to_dict() == PICK_B.to_dict()


def test_leaving_the_fight_returns_control_to_ppo():
    wrapped, lane, search = agent()
    wrapped.choose_action(obs(fight("elite")))
    wrapped.choose_action(obs(MAP))
    assert lane.calls[-1] == ("ppo", "map")


def test_searched_decisions_are_recorded_with_their_visit_distribution(tmp_path):
    wrapped, lane, search = agent(tmp_path)
    wrapped.choose_action(obs(fight("elite")))
    wrapped.choose_action(obs(HAND_SELECT))  # answered from the tree: not a searched decision
    with gzip.open(tmp_path / "decisions.jsonl.gz", "rt", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    assert len(lines) == 1
    line = lines[0]
    assert line["candidates"] == [c.to_dict() for c in (STRIKE, DEFEND, END)]
    assert line["expert_index"] == 1
    assert line["target_distribution"] == [0.1, 0.8, 0.1]
    assert line["state_type"] == "elite"


def test_the_wrapper_passes_learning_calls_through():
    wrapped, lane, search = agent()
    transition = Transition(state=obs(MAP), action=NODE, reward=1.0, next_state=obs(MAP), done=False)
    wrapped.observe(transition)
    wrapped.finish_episode(obs(MAP), truncated=True)
    assert lane.calls[-2:] == [("observe", None), ("finish", True)]

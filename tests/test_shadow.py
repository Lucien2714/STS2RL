"""A simulator following a real client: mirroring, agreement, and giving the copy up."""

from __future__ import annotations

import pytest

from sts2rl.actions import GameAction
from sts2rl.env.mcp_client import STS2ClientError
from sts2rl.env.reset import ResetSpec
from sts2rl.env.types import EnvStep, GameObservation
from sts2rl.search import SearchCombatAgent
from sts2rl.search.shadow import ShadowEnv, fingerprint
from sts2rl.training.eval_cli import _seed_pools, _shadow_urls, create_parser

from test_search_agent import FakeLane, FakeSearch

CARD = GameAction("select_card_reward", card_index=0)
PROCEED = GameAction("proceed")
NODE = GameAction("choose_map_node", index=0)
STRIKE = GameAction("play_card", card_index=0, target="JAW_WORM_0")


def combat(hp: int = 40) -> dict:
    return {
        "state_type": "monster",
        "run": {"act": 1, "floor": 2},
        "player": {"hp": 80, "max_hp": 80, "energy": 3, "hand": [{"id": "STRIKE_IRONCLAD"}]},
        "battle": {"round": 1, "enemies": [{"entity_id": "JAW_WORM_0", "hp": hp, "intents": []}]},
    }


REWARDS = {"state_type": "rewards", "run": {"floor": 2}, "rewards": {"items": []}}
MAP = {"state_type": "map", "run": {"floor": 2}, "map": {"next_options": [{"col": 1, "row": 3}]}}
EVENT = {"state_type": "event", "run": {"floor": 1}, "event": {"options": [{"text_key": "A"}]}}


class ScriptedEnv:
    """Answers each action from a table; refuses anything the table does not hold."""

    def __init__(self, backend: str, first: dict, moves: dict, reused: bool = False) -> None:
        self.backend = backend
        self.first = first
        self.moves = moves
        self.state = first
        self.reused_active_run = reused
        self.sent: list[str] = []
        self.resets: list[ResetSpec] = []
        self.client = object()
        self.fail_with: Exception | None = None

    def reset(self, spec=None):
        self.resets.append(spec)
        self.state = self.first
        return self.state

    def step(self, action):
        if self.fail_with is not None:
            raise self.fail_with
        self.sent.append(action.action_type)
        nxt = self.moves.get((self.state["state_type"], action.action_type))
        if nxt is None:
            return EnvStep(raw_state=self.state, done=False, info={"action_error": True})
        self.state = nxt
        return EnvStep(raw_state=nxt, done=False, info={"action_error": False})

    def get_state(self):
        return self.state

    def get_player_detail(self):
        return None

    def close(self):
        pass


def pair(real_moves, sim_moves, first=None, **real_kwargs):
    first = first or combat()
    real = ScriptedEnv("game", first, real_moves, **real_kwargs)
    sim = ScriptedEnv("sim", first, sim_moves)
    return ShadowEnv(real, sim), real, sim


SEEDED = ResetSpec(game_mode="custom", run_seed="3EH64E978B")


def test_the_simulator_starts_the_same_seed_and_character():
    shadow, real, sim = pair({}, {})
    shadow.reset(ResetSpec(game_mode="custom", run_seed="3EH64E978B", character=0, ascension=2))
    assert shadow.alive
    started = sim.resets[0]
    assert (started.run_seed, started.character, started.ascension) == ("3EH64E978B", 0, 2)


def test_an_accepted_action_is_played_on_both_and_the_states_agree():
    shadow, real, sim = pair({("monster", "play_card"): combat(34)}, {("monster", "play_card"): combat(34)})
    shadow.reset(SEEDED)
    assert shadow.agrees(combat())
    step = shadow.step(STRIKE)
    assert step.raw_state is real.state
    assert sim.sent == ["play_card"]
    assert shadow.agrees(step.raw_state)
    assert shadow.stats.agreed == 2


def test_an_action_the_real_game_refused_is_not_sent_to_the_simulator():
    shadow, real, sim = pair({}, {("monster", "play_card"): combat(34)})
    shadow.reset(SEEDED)
    shadow.step(STRIKE)
    assert sim.sent == []


def test_a_screen_the_simulator_skipped_disagrees_once_and_then_agrees_again():
    # The real client stops on the reward screen after the card; the simulator is
    # already on the map, refuses the proceed, and the next decision agrees.
    real_moves = {("card_reward", "select_card_reward"): REWARDS, ("rewards", "proceed"): MAP}
    sim_moves = {("card_reward", "select_card_reward"): MAP}
    shadow, real, sim = pair(real_moves, sim_moves, first={"state_type": "card_reward", "run": {"floor": 2}})
    shadow.reset(SEEDED)
    shadow.step(CARD)
    assert not shadow.agrees(real.state)
    shadow.step(PROCEED)
    assert shadow.stats.refused == 1
    assert shadow.agrees(real.state)
    assert shadow.alive


def test_a_run_of_disagreements_gives_the_copy_up_for_the_episode():
    shadow, real, sim = pair({("monster", "play_card"): combat(34)}, {("monster", "play_card"): combat(30)})
    shadow.reset(SEEDED)
    shadow.step(STRIKE)
    for _ in range(3):
        assert not shadow.agrees(real.state)
    assert not shadow.alive
    assert shadow.stats.lost_at["reason"] == "3 decisions in a row disagreed"
    assert shadow.stats.differences["enemies"]["real"][0][1] == 34
    assert shadow.stats.differences["enemies"]["sim"][0][1] == 30
    # Agreeing again later does not bring it back, and nothing more is mirrored.
    sim.state = real.state
    assert not shadow.agrees(real.state)
    shadow.step(STRIKE)
    assert sim.sent == ["play_card"]


def test_a_joined_run_a_modifier_or_a_missing_seed_is_never_followed():
    shadow, _, sim = pair({}, {}, reused=True)
    shadow.reset(SEEDED)
    assert not shadow.alive and "joined" in shadow.stats.lost_at["reason"]
    for spec in (ResetSpec(game_mode="custom", run_seed="X", modifiers=("FLIGHT",)), ResetSpec()):
        shadow, _, sim = pair({}, {})
        shadow.reset(spec)
        assert not shadow.alive and sim.resets == []


def test_a_failing_simulator_costs_the_search_not_the_episode():
    shadow, real, sim = pair({("monster", "play_card"): combat(34)}, {})
    shadow.reset(SEEDED)
    sim.fail_with = STS2ClientError("connection refused")
    step = shadow.step(STRIKE)
    assert step.raw_state["battle"]["enemies"][0]["hp"] == 34
    assert not shadow.alive


def test_each_episode_keeps_its_own_record():
    shadow, real, sim = pair({}, {})
    shadow.reset(SEEDED)
    shadow.agrees(real.state)
    shadow.reset(SEEDED)
    records = shadow.finish()
    assert [record["agreed"] for record in records] == [1, 0]


def test_a_failed_reset_opens_no_record():
    shadow, real, sim = pair({}, {})
    real.reset = lambda spec=None: (_ for _ in ()).throw(STS2ClientError("menu raced"))
    with pytest.raises(STS2ClientError):
        shadow.reset(SEEDED)
    del real.reset
    shadow.reset(SEEDED)
    shadow.agrees(real.state)
    assert [record["agreed"] for record in shadow.finish()] == [1]


def test_shop_items_are_told_apart_by_kind_price_and_stock():
    def shop(*items):
        return {"state_type": "shop", "shop": {"items": list(items)}}

    relic = {"category": "relic", "relic_id": "VAJRA", "price": 150, "is_stocked": True}
    removal = {"category": "card_removal", "price": 75, "is_stocked": True}
    assert fingerprint(shop(relic, removal)) == fingerprint(shop(dict(relic), dict(removal)))
    assert fingerprint(shop(relic)) != fingerprint(shop({**relic, "relic_id": "ANCHOR"}))
    assert fingerprint(shop(removal)) != fingerprint(shop({**removal, "price": 100}))
    assert fingerprint(shop(removal)) != fingerprint(shop({**removal, "is_stocked": False}))


def test_display_text_does_not_count_as_a_disagreement():
    worded = combat()
    worded["battle"]["enemies"][0]["intents"] = [{"type": "Attack", "label": "11", "description": "Deal 11."}]
    reworded = combat()
    reworded["battle"]["enemies"][0]["intents"] = [{"type": "Attack", "label": "11", "description": "11 dmg"}]
    assert fingerprint(worded) == fingerprint(reworded)


def test_a_fight_the_copy_does_not_match_is_played_by_the_actor():
    lane, search = FakeLane(), FakeSearch()
    answers = iter([True, False])
    wrapped = SearchCombatAgent(lane, env=object(), search=search, gate=lambda raw: next(answers))
    wrapped.reset(GameObservation(MAP))
    wrapped.choose_action(GameObservation(combat()))
    wrapped.choose_action(GameObservation(combat()))
    assert search.calls == 1
    assert wrapped.unsearchable == 1
    assert [call[0] for call in lane.calls if call[0] != "reset"] == ["external", "ppo"]


def test_the_gate_is_asked_once_per_decision_outside_fights_too():
    asked: list[str] = []
    lane, search = FakeLane(), FakeSearch()
    wrapped = SearchCombatAgent(lane, env=object(), search=search, gate=lambda raw: asked.append(raw["state_type"]) or True)
    wrapped.reset(GameObservation(MAP))
    wrapped.choose_action(GameObservation(MAP))
    wrapped.choose_action(GameObservation(combat()))
    assert asked == ["map", "monster"]


def parse(*argv: str):
    return create_parser().parse_args(["--run-dir", "runs/x", *argv])


def test_shadow_sims_need_a_real_backend_the_search_and_one_per_client():
    assert _shadow_urls(parse(), "game", 1) == ()
    ok = parse("--search-combat", "--shadow-sims", "http://localhost:15620/api/v1")
    assert _shadow_urls(ok, "game", 1) == ("http://localhost:15620/api/v1",)
    with pytest.raises(ValueError, match="--backend game"):
        _shadow_urls(ok, "sim", 1)
    with pytest.raises(ValueError, match="--search-combat"):
        _shadow_urls(parse("--shadow-sims", "http://a"), "game", 1)
    with pytest.raises(ValueError, match="one simulator per client"):
        _shadow_urls(ok, "game", 2)
    with pytest.raises(ValueError, match="repeats"):
        _shadow_urls(parse("--search-combat", "--shadow-sims", "http://a,http://a"), "game", 2)


class Pools:
    training_seeds = ("AAA", "BBB", "CCC")
    holdout_seeds = ("HHH",)


def test_only_seeds_plays_those_seeds_from_whichever_pool_holds_them():
    assert _seed_pools(parse("--only-seeds", "BBB,HHH"), Pools) == (("BBB",), ("HHH",))
    with pytest.raises(ValueError, match="ZZZ"):
        _seed_pools(parse("--only-seeds", "BBB,ZZZ"), Pools)
    with pytest.raises(ValueError, match="HHH"):
        _seed_pools(parse("--only-seeds", "HHH", "--pools", "training"), Pools)

"""The combat search on small games with a known answer, and its leaf evaluator."""

from __future__ import annotations

import math
import random

import pytest

from sts2rl.actions import GameAction
from sts2rl.search import CombatSearch, LeafEvaluator, MctsConfig, action_key, fit_weights
from sts2rl.search.evaluate import features, intent_damage

GOOD, BAD, RISKY, RARE, END = (GameAction(name) for name in ("good", "bad", "risky", "rare", "end_turn"))


class BranchingGame:
    """A fight of two turns whose hidden part is the world the seed picks.

    ``risky`` pays +4 in even worlds and -10 in odd ones: planned against one
    world it is the best action, averaged over worlds it is the worst. ``rare``
    is only offered in worlds divisible by three.
    """

    def __init__(self, world: int = 0) -> None:
        self.state = {"turns": 0, "score": 0.0, "world": world}
        self.points: dict[int, dict] = {}
        self.reseeds: list[int] = []
        self.released: list[int] = []

    def snapshot(self) -> int:
        point = len(self.points) + 1
        self.points[point] = dict(self.state)
        return point

    def restore(self, point: int) -> dict:
        self.state = dict(self.points[point])
        return self.state

    def release(self, point: int) -> None:
        self.released.append(point)

    def reseed(self, seed: int) -> None:
        self.reseeds.append(seed)
        self.state["world"] = seed

    def step(self, action: GameAction) -> dict:
        state = dict(self.state)
        if action.action_type == "end_turn":
            state["turns"] += 1
        elif action.action_type == "good":
            state["score"] += 2
        elif action.action_type == "risky":
            state["score"] += 4 if state["world"] % 2 == 0 else -10
        elif action.action_type == "rare":
            state["score"] += 3
        self.state = state
        return state


def candidates(state: dict) -> list[GameAction]:
    offered = [GOOD, BAD, RISKY, END]
    if state["world"] % 3 == 0:
        offered.insert(0, RARE)
    return offered


def evaluate(state: dict, last_combat: dict | None = None) -> float:
    return 1 / (1 + math.exp(-state["score"] / 4))


def search_for(game: BranchingGame, *, reseed: bool = True, **config) -> CombatSearch:
    search = CombatSearch(
        MctsConfig(**{"simulations": 400, "turn_depth": 1, **config}),
        evaluator=evaluate,
        candidates=candidates,
        key=lambda state, action: (action.action_type,),
        rng=random.Random(0),
    )
    if not reseed:
        game.reseed = lambda seed: game.reseeds.append(seed)  # the world never changes
    return search


@pytest.fixture(autouse=True)
def _fight_ends_with_the_depth(monkeypatch):
    # The toy game has no fight-over state; the depth ends every simulation.
    monkeypatch.setattr("sts2rl.search.mcts.is_fight_over", lambda state: False)


def test_reseeding_averages_over_worlds_instead_of_planning_against_one():
    game = BranchingGame(world=0)
    clairvoyant = search_for(game, reseed=False).search(game, game.state)
    assert clairvoyant.action == RISKY

    game = BranchingGame(world=0)
    fair = search_for(game).search(game, game.state)
    assert fair.action in (GOOD, RARE)
    assert fair.action != RISKY


def test_a_clairvoyant_search_never_reseeds_and_plans_against_the_real_world():
    game = BranchingGame(world=0)
    search = search_for(game)
    search.config = MctsConfig(simulations=400, turn_depth=1, reseed=False)
    result = search.search(game, game.state)
    assert game.reseeds == []
    assert result.action == RISKY  # +4 in the world it can see


def test_the_seed_pool_is_cycled_and_fresh_per_decision():
    game = BranchingGame()
    search = search_for(game, simulations=64, seed_pool=8)
    search.search(game, game.state)
    first = game.reseeds[:]
    assert len(first) == 64 and len(set(first)) == 8
    assert first[:8] == first[8:16]

    search.search(game, game.state)
    assert set(game.reseeds[64:]).isdisjoint(first)


def test_availability_counts_only_the_worlds_that_offered_an_action():
    game = BranchingGame()
    result = search_for(game, simulations=300).search(game, game.state)
    root = result.root
    assert root.available[("good",)] == 300
    assert 0 < root.available[("rare",)] < 300


def test_widening_limits_how_many_actions_a_node_has_tried():
    """The limit counts the tried actions offered now; one missing from this world frees a slot."""
    game = BranchingGame()
    search = search_for(game, simulations=3, widening_c=1.0, widening_alpha=0.5)
    search.candidates = lambda state: [GOOD, BAD, RISKY, END]
    result = search.search(game, game.state)
    assert len(result.root.children) <= math.ceil(1.0 * 3**0.5)


def test_small_value_differences_still_concentrate_the_budget():
    """Leaves a thousandth apart: normalised against the tree's range, the best still wins out."""
    game = BranchingGame()
    search = search_for(game, simulations=200)
    search.evaluate = lambda state, last=None: 0.5 + 0.001 * state["score"]
    search.candidates = lambda state: [GOOD, BAD, END]
    result = search.search(game, game.state)
    assert result.action == GOOD
    assert result.distribution[0] > 0.5


def test_the_branch_point_is_released_even_when_the_search_fails():
    game = BranchingGame()

    def broken(state: dict, last_combat: dict | None = None) -> float:
        raise RuntimeError("evaluator failed")

    search = CombatSearch(MctsConfig(simulations=4, turn_depth=1), evaluator=broken, candidates=candidates)
    with pytest.raises(RuntimeError):
        search.search(game, game.state)
    assert game.released == [1]


def test_copies_of_one_card_are_one_decision_and_split_its_share():
    hand = [
        {"index": 0, "id": "STRIKE_IRONCLAD", "current_upgrade_level": 0},
        {"index": 1, "id": "STRIKE_IRONCLAD", "current_upgrade_level": 0},
        {"index": 2, "id": "STRIKE_IRONCLAD", "current_upgrade_level": 1},
    ]
    state = {"player": {"hand": hand}}
    first, second, upgraded = (
        GameAction("play_card", card_index=i, target="BOSS_0") for i in range(3)
    )
    assert action_key(state, first) == action_key(state, second)
    assert action_key(state, first) != action_key(state, upgraded)
    assert action_key(state, first) != action_key(state, GameAction("play_card", card_index=0, target="MINION_1"))


def _combat(hp=50, block=0, enemy_hp=100, intents=(("Attack", "10"),), status=(), enemy_status=(), potions=1):
    return {
        "state_type": "boss",
        "player": {
            "hp": hp, "max_hp": 80, "block": block, "max_potion_slots": 3,
            "status": [{"id": i, "amount": a, "type": "Buff"} for i, a in status],
            "potions": [{"id": "FIRE_POTION", "slot": s} for s in range(potions)],
        },
        "battle": {"enemies": [{
            "hp": enemy_hp, "max_hp": 200, "block": 0,
            "status": [{"id": i, "amount": a, "type": "Debuff"} for i, a in enemy_status],
            "intents": [{"type": t, "label": label} for t, label in intents],
        }]},
    }


def test_the_evaluator_prefers_health_progress_and_growth():
    value = LeafEvaluator()
    assert value(_combat(hp=60)) > value(_combat(hp=30))
    assert value(_combat(enemy_hp=40)) > value(_combat(enemy_hp=160))
    assert value(_combat(block=15)) > value(_combat(block=0))
    assert value(_combat(status=[("STRENGTH_POWER", 3)])) > value(_combat())
    assert value(_combat(enemy_status=[("VULNERABLE_POWER", 2)])) > value(_combat())


def test_a_finished_fight_is_scored_exactly():
    value = LeafEvaluator()
    won = {"state_type": "rewards", "player": {"hp": 40, "max_hp": 80, "potions": [], "max_potion_slots": 3}}
    healthier = {**won, "player": {**won["player"], "hp": 70}}
    lost = {"state_type": "game_over", "terminal_reason": "death", "player": {"hp": 0, "max_hp": 80}}
    assert value(lost) == 0.0
    assert 0.7 < value(won) < value(healthier) <= 1.0
    # A loss earns a little for the damage dealt before it, never as much as a win.
    closer, farther = _combat(enemy_hp=20), _combat(enemy_hp=180)
    assert value(lost, farther) < value(lost, closer) < 0.7


def test_a_fight_that_looks_lost_still_prefers_the_line_that_hurts_the_enemy():
    value = LeafEvaluator()
    hopeless = dict(hp=3, intents=(("Attack", "40"),))
    assert value(_combat(enemy_hp=150, **hopeless)) > value(_combat(enemy_hp=190, **hopeless))


def test_intent_damage_reads_single_and_multi_hit_labels():
    assert intent_damage({"type": "Attack", "label": "11"}) == 11
    assert intent_damage({"type": "Attack", "label": "7x2"}) == 14
    # A status card intent's number counts cards, not damage.
    assert intent_damage({"type": "StatusCard", "label": "3"}) == 0
    assert intent_damage({"type": "Buff", "label": ""}) == 0


def test_lethal_incoming_is_flagged():
    assert features(_combat(hp=8, intents=(("Attack", "5x2"),)))["lethal_incoming"] == 1.0
    assert features(_combat(hp=8, block=5, intents=(("Attack", "5x2"),)))["lethal_incoming"] == 0.0


def test_fitting_recovers_the_direction_of_a_separable_signal():
    rng = random.Random(1)
    rows, won = [], []
    for _ in range(400):
        hp = rng.random()
        rows.append({"bias": 1.0, "hp_ratio": hp})
        won.append(hp + rng.gauss(0, 0.1) > 0.5)
    weights = fit_weights(rows, won)
    assert weights["hp_ratio"] > 1.0


def test_the_environment_is_left_at_the_decision_with_its_own_world():
    """The last simulation's world is not the game's; the real action is taken from the decision."""
    game = BranchingGame(world=4)
    game.state["score"] = 1.0
    search_for(game, simulations=20).search(game, game.state)
    assert game.state == {"turns": 0, "score": 1.0, "world": 4}


def test_a_time_budget_stops_the_search_early_but_always_simulates_once():
    game = BranchingGame()
    result = search_for(game, simulations=100_000, seconds=0.0).search(game, game.state)
    assert result.root.visits == 1


def test_deck_output_reads_damage_block_and_hits_from_descriptions():
    from sts2rl.search.evaluate import deck_output

    cards = [
        {"description": "Deal 6 damage.", "cost": "1"},
        {"description": "Deal 5 damage twice.", "cost": "1"},
        {"description": "Deal 3 damage to a random enemy 3 times.", "cost": "1"},
        {"description": "Deal 5 damage to ALL enemies X times.", "cost": "X"},
        {"description": "Gain 5 Block.", "cost": "1", "quantity": 2},
        {"description": "Unplayable.", "cost": "-1"},
    ]
    damage, block = deck_output(cards, energy=3.0)
    # 6 + 10 + 9 + 15 damage and 10 block over 1+1+1+3+2+1 energy.
    assert damage == pytest.approx(40 / 9)
    assert block == pytest.approx(10 / 9)


def test_the_race_favours_the_deck_that_kills_faster():
    weak = _combat()
    strong = _combat()
    weak["player"]["draw_pile"] = [{"description": "Deal 4 damage.", "cost": "1"}] * 10
    strong["player"]["draw_pile"] = [{"description": "Deal 12 damage.", "cost": "1"}] * 10
    assert features(strong)["race"] > features(weak)["race"]
    assert features(strong)["deck_damage"] > features(weak)["deck_damage"]
    assert LeafEvaluator()(strong) > LeafEvaluator()(weak)


def test_a_signed_fit_keeps_each_weight_on_its_side_of_zero():
    from sts2rl.search.evaluate import FEATURE_SIGNS

    # Enemy Vulnerable appears only in lost fights: a free fit makes it a penalty.
    rows = [{"bias": 1.0, "enemy_vulnerable": 1.0, "hp_ratio": 0.2}] * 20 + [{"bias": 1.0, "hp_ratio": 0.9}] * 20
    won = [False] * 20 + [True] * 20
    assert fit_weights(rows, won, signs=None)["enemy_vulnerable"] < 0
    signed = fit_weights(rows, won)
    assert signed["enemy_vulnerable"] == 0.0
    assert signed["hp_ratio"] > 0
    assert set(FEATURE_SIGNS) <= set(signed)

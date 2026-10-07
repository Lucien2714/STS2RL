"""The combat search on small games with a known answer, and its leaf evaluator."""

from __future__ import annotations

import math
import random

import pytest

from sts2rl.actions import GameAction
from sts2rl.search import CombatSearch, LeafEvaluator, MctsConfig, action_key, fit_weights
from sts2rl.search.evaluate import end_of_turn_damage, features, intent_damage, pending_summons

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
        self.restores: list[tuple[int, int | None]] = []
        self.released: list[int] = []

    def snapshot(self) -> int:
        point = len(self.points) + 1
        self.points[point] = dict(self.state)
        return point

    def restore(self, point: int, seed: int | None = None) -> dict:
        self.restores.append((point, seed))
        self.state = dict(self.points[point])
        if seed is not None:
            self.reseed(seed)
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


def evaluate(state: dict, last_combat: dict | None = None, root: dict | None = None) -> float:
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


def test_each_simulation_restores_and_reseeds_in_one_call():
    """The seed rides on the restore (one simulator request), and the final restore,
    which leaves the game where the decision was taken, carries none."""
    game = BranchingGame()
    search = search_for(game, simulations=5, seed_pool=8)
    search.search(game, game.state)
    seeded, final = game.restores[:-1], game.restores[-1]
    assert [seed for _, seed in seeded] == game.reseeds
    assert len(seeded) == 5 and None not in game.reseeds
    assert final == (1, None)

    game = BranchingGame()
    search = search_for(game, simulations=5, reseed=True)
    search.config = MctsConfig(simulations=5, turn_depth=1, reseed=False)
    search.search(game, game.state)
    assert [seed for _, seed in game.restores] == [None] * 6
    assert game.reseeds == []


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
    search.evaluate = lambda state, last=None, root=None: 0.5 + 0.001 * state["score"]
    search.candidates = lambda state: [GOOD, BAD, END]
    result = search.search(game, game.state)
    assert result.action == GOOD
    assert result.distribution[0] > 0.5


def test_an_excluded_root_action_is_neither_chosen_nor_searched():
    """The simulator may accept what the game refused; the search must not pick it again."""
    game = BranchingGame(world=0)
    assert search_for(game, reseed=False).search(game, game.state).action == RISKY

    game = BranchingGame(world=0)
    result = search_for(game, reseed=False).search(game, game.state, exclude=[RISKY])
    assert result.action != RISKY
    assert RISKY not in result.candidates
    assert ("risky",) not in result.root.children
    assert ("risky",) not in result.root.available


def test_an_excluded_copy_leaves_the_other_copy_standing_for_its_key():
    """Two copies of one card are one decision: excluding one keeps the key."""
    first, second = GameAction("good", copy=1), GameAction("good", copy=2)
    game = BranchingGame(world=0)
    search = search_for(game, reseed=False, simulations=100)
    search.candidates = lambda state: [first, second, BAD, END]
    result = search.search(game, game.state, exclude=[first])
    assert result.candidates == (second, BAD, END)
    assert result.action is second
    assert ("good",) in result.root.children


def test_no_exclusion_searches_exactly_as_before():
    runs = []
    for exclude in (None, ()):
        game = BranchingGame(world=0)
        search = search_for(game, simulations=100)
        result = search.search(game, game.state) if exclude is None else search.search(game, game.state, exclude=exclude)
        runs.append((result.action, result.visits, result.values, game.reseeds))
    assert runs[0] == runs[1]


def test_the_branch_point_is_released_even_when_the_search_fails():
    game = BranchingGame()

    def broken(state: dict, last_combat: dict | None = None, root: dict | None = None) -> float:
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


def _killed_giant(hp=50, block=0, eruption=30):
    """Waterfall Giant after the killing blow: a placeholder HP until it erupts."""
    state = _combat(hp=hp, block=block)
    state["battle"]["enemies"] = [{
        "name": "Waterfall Giant", "hp": 999_999_999, "max_hp": 999_999_999, "block": 0,
        "status": [{"id": "STEAM_ERUPTION_POWER", "amount": eruption, "type": "Buff"}], "intents": [],
    }]
    return state


def test_killing_waterfall_giant_is_progress_not_a_boss_at_full_health():
    """The killed Giant reads 999,999,999 of 999,999,999 HP; the search avoided the kill."""
    value = LeafEvaluator()
    almost_dead = _combat(enemy_hp=10)
    almost_dead["battle"]["enemies"][0]["max_hp"] = 240
    killed = _killed_giant(eruption=10)  # the same threat as the living Giant's attack
    assert features(killed)["enemy_hp_ratio"] == 0.0
    assert value(killed) > value(almost_dead)


def test_the_eruption_of_a_killed_giant_counts_as_incoming_damage():
    assert features(_killed_giant(hp=20, eruption=30))["lethal_incoming"] == 1.0
    assert features(_killed_giant(hp=20, block=15, eruption=30))["lethal_incoming"] == 0.0
    assert LeafEvaluator()(_killed_giant(hp=20, block=15)) > LeafEvaluator()(_killed_giant(hp=20))


def _fight(*enemies, hp=50):
    """A fight against ``enemies``: (entity id, hp, max hp, powers, intents) each."""
    state = _combat(hp=hp)
    state["battle"]["enemies"] = [
        {
            "entity_id": eid, "name": eid, "hp": e_hp, "max_hp": e_max, "block": 0,
            "status": [{"id": i, "amount": a, "type": "Buff"} for i, a in powers],
            "intents": [{"type": t, "label": label} for t, label in intents],
        }
        for eid, e_hp, e_max, powers, intents in enemies
    ]
    return state


def test_killing_one_of_several_enemies_is_progress():
    """The game drops a killed enemy from the list; against the enemies still standing the
    kill raised the share of health left (22/40 -> 20/20), so the search scored it as a
    setback. Against the search root's enemies it is the progress it is."""
    value = LeafEvaluator()
    root = _fight(("SLIME_0", 2, 20, (), ()), ("SLIME_1", 20, 20, (), ()))
    killed = _fight(("SLIME_1", 20, 20, (), ()))
    assert features(killed, root)["enemy_hp_ratio"] < features(root, root)["enemy_hp_ratio"]
    assert value(killed, root=root) > value(root, root=root)


def test_the_search_root_changes_nothing_when_no_enemy_left_or_arrived():
    root = _fight(("SLIME_0", 12, 20, (), ()), ("SLIME_1", 20, 20, (), ()))
    hurt = _fight(("SLIME_0", 4, 20, (), ()), ("SLIME_1", 20, 20, (), ()))
    assert features(root, root) == features(root)
    assert features(hurt, root) == features(hurt)


def test_a_summoner_stands_for_what_it_summons_on_death():
    """Gremlin Merc summons two gremlins when it dies; read as fresh enemies at full health,
    they made the nearly dead Merc the better position and the search held back the kill."""
    value = LeafEvaluator()
    merc = _fight(("GREMLIN_MERC_0", 5, 48, (("SURPRISE_POWER", 1),), (("Attack", "8x2"),)))
    after = _fight(("FAT_GREMLIN_0", 16, 16, (), ()), ("SNEAKY_GREMLIN_0", 13, 13, (), (("Attack", "9"),)))
    assert features(after, merc)["enemy_hp_ratio"] < features(merc, merc)["enemy_hp_ratio"]
    assert value(after, root=merc) > value(merc, root=merc)
    # Without the summons in the totals, a Merc at full health would look like a short fight.
    plain = _fight(("GREMLIN_MERC_0", 48, 48, (), (("Attack", "8x2"),)))
    summoner = _fight(("GREMLIN_MERC_0", 48, 48, (("SURPRISE_POWER", 1),), (("Attack", "8x2"),)))
    assert features(summoner)["enemy_hp_ratio"] == 1.0
    assert features(summoner)["race"] < features(plain)["race"]


def test_an_infested_parasite_stands_for_each_wriggler():
    parasite = _fight(("PHROG_PARASITE_0", 3, 63, (("INFESTED_POWER", 4),), (("Attack", "4x4"),)))
    wrigglers = _fight(*[(f"WRIGGLER_{i}", 19, 19, (), ()) for i in range(4)])
    assert pending_summons(parasite["battle"]["enemies"][0]) == 4 * 19.0
    assert features(wrigglers, parasite)["enemy_hp_ratio"] < features(parasite, parasite)["enemy_hp_ratio"]
    assert LeafEvaluator()(wrigglers, root=parasite) > LeafEvaluator()(parasite, root=parasite)


def _held(card_id, description, can_play=False):
    return {"id": card_id, "type": "Status", "description": description, "can_play": can_play}


INFECTION = _held("INFECTION", "Unplayable. At the end of your turn, if this is in your Hand, take 3 damage.")


def test_an_unplayable_card_that_hurts_at_end_of_turn_is_incoming_damage():
    assert end_of_turn_damage(INFECTION) == 3.0
    assert end_of_turn_damage(_held("BAD_LUCK", "At the end of your turn, if this is in your Hand, lose 13 HP.")) == 13.0
    # A card the player can still play is a choice the search makes, not certain damage.
    toxic = _held("TOXIC", "At the end of your turn, if this is in your Hand, take 5 damage. Exhaust.", can_play=True)
    assert end_of_turn_damage(toxic) == 0.0
    assert end_of_turn_damage(_held("DAZED", "Unplayable. Ethereal.")) == 0.0
    # Two Infections turn a survivable hit into a lethal one.
    state = _combat(hp=15, intents=(("Attack", "10"),))
    assert features(state)["lethal_incoming"] == 0.0
    state["player"]["hand"] = [INFECTION, INFECTION]
    assert features(state)["lethal_incoming"] == 1.0
    assert LeafEvaluator()(state) < LeafEvaluator()(_combat(hp=15, intents=(("Attack", "10"),)))


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


def test_a_budget_given_with_the_decision_replaces_the_configured_one():
    game = BranchingGame()
    search = search_for(game, simulations=400)
    assert search.search(game, game.state, simulations=7).root.visits == 7
    assert search.search(game, game.state).root.visits == 400


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

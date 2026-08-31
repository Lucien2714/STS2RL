from sts2rl.env.player import Player


def test_player_detail_update_ignores_non_scalar_int_values():
    player = Player(character=0)
    player.update_from_detail(
        {
            "state_type": "player_detail",
            "status": "ok",
            "player": {
                "character": "The Ironclad",
                "hp": {"unexpected": "object"},
                "max_hp": object(),
                "block": [],
                "gold": "44",
                "max_potion_slots": None,
                "status": [],
                "relics": [],
                "potions": [],
                "deck_count": {"bad": "count"},
                "deck": [
                    {
                        "id": "WHIRLWIND",
                        "name": "Whirlwind",
                        "cost": "X",
                        "star_cost": None,
                        "is_upgraded": False,
                    },
                ],
            },
        }
    )

    assert player.current_hp == 80
    assert player.max_hp == 80
    assert player.block == 0
    assert player.gold == 44
    assert player.deck_count == 1
    assert player.current_deck[0].regular_cost == -1
    assert player.current_deck[0].star_cost == 0

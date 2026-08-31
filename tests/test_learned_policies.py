"""The learned-policy variant trains its card embedding end to end (ADR-0007 Phase 3)."""

import pytest
import torch

from sts2rl.agents.orchestrator import create_battle_agent
from sts2rl.encoders.battle_encoder import BattleStateEncoder
from sts2rl.encoders.learned_battle_encoder import (
    CARD_FACTOR_CHANNELS,
    LearnedBattleStateEncoder,
)

BATTLE_STATE = {
    "state_type": "monster",
    "in_battle": True,
    "battle": {
        "turn": "player",
        "is_play_phase": True,
        "enemies": [{"entity_id": "JAW_WORM_0", "id": "JAW_WORM", "hp": 40, "max_hp": 44}],
    },
    "player": {
        "hp": 70,
        "max_hp": 80,
        "energy": 3,
        "max_energy": 3,
        "block": 0,
        "hand": [
            {"index": 0, "id": "STRIKE_IRONCLAD", "cost": 1, "type": "attack", "target_type": "enemy"},
            {"index": 1, "id": "DEFEND_IRONCLAD", "cost": 1, "type": "skill", "target_type": "self"},
        ],
        "potions": [],
        "status": [],
        "relics": [],
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
    },
}


def test_learned_encoder_appends_index_channels_to_the_flat_layout():
    """The learned layout is the handcrafted one plus trailing index columns."""
    flat = BattleStateEncoder()
    learned = LearnedBattleStateEncoder()

    assert learned.state_size == flat.state_size + learned.MAX_HAND * CARD_FACTOR_CHANNELS
    assert learned.action_feature_size == flat.action_feature_size + CARD_FACTOR_CHANNELS
    assert learned.state_feature_size == flat.state_size
    # The handcrafted prefix must be untouched, so the added channels are purely
    # additional information rather than a different featurization.
    assert learned.encode_state(BATTLE_STATE)[: flat.state_size] == flat.encode_state(BATTLE_STATE)


def test_learned_encoder_writes_real_card_indices():
    """Occupied hand slots carry a non-zero card row; empty slots stay zero."""
    learned = LearnedBattleStateEncoder()
    features = learned.encode_state(BATTLE_STATE)
    indices = features[learned.state_card_index_offset :]

    first_card_row = indices[0]
    second_card_row = indices[CARD_FACTOR_CHANNELS]
    empty_slot_row = indices[2 * CARD_FACTOR_CHANNELS]

    assert first_card_row > 0
    assert second_card_row > 0
    assert first_card_row != second_card_row  # Strike and Defend are different rows
    assert empty_slot_row == 0


@pytest.mark.parametrize("agent_type", ["DQN", "PPO"])
def test_card_embedding_is_trained_by_the_agent_optimizer(agent_type):
    """The whole point: the embedding is in the optimizer and receives gradient."""
    agent = create_battle_agent(agent_type, policy="learned")
    embedding = agent.model.featurizer.card_encoder.id_emb.weight

    optimized = {id(p) for group in agent.optimizer.param_groups for p in group["params"]}
    assert id(embedding) in optimized

    state = agent.encode_state(BATTLE_STATE)
    rows = torch.tensor(
        [state + vector for vector in agent.candidate_action_vectors(BATTLE_STATE)],
        dtype=torch.float32,
        device=agent.device,
    )
    agent.bc_score(rows).sum().backward()

    assert embedding.grad is not None
    assert float(embedding.grad.abs().sum()) > 0


@pytest.mark.parametrize("agent_type", ["DQN", "PPO"])
def test_learned_and_flat_checkpoints_are_mutually_incompatible(agent_type, tmp_path):
    """Different schemas must stop the two variants from loading each other."""
    learned = create_battle_agent(agent_type, policy="learned")
    flat = create_battle_agent(agent_type)
    assert learned.ACTION_SCHEMA != flat.ACTION_SCHEMA

    learned_path = tmp_path / "learned.pt"
    flat_path = tmp_path / "flat.pt"
    learned.save(str(learned_path))
    flat.save(str(flat_path))

    with pytest.raises(ValueError):
        create_battle_agent(agent_type, policy="learned").load(str(flat_path))
    with pytest.raises(ValueError):
        create_battle_agent(agent_type).load(str(learned_path))


@pytest.mark.parametrize("agent_type", ["DQN", "PPO"])
def test_learned_checkpoint_round_trips_the_embedding(agent_type, tmp_path):
    """Saving and loading must preserve the learned card vectors."""
    agent = create_battle_agent(agent_type, policy="learned")
    with torch.no_grad():
        agent.model.featurizer.card_encoder.id_emb.weight.add_(1.0)
    weights = agent.model.featurizer.card_encoder.id_emb.weight.clone()

    path = tmp_path / "agent.pt"
    agent.save(str(path))
    restored = create_battle_agent(agent_type, policy="learned")
    restored.load(str(path))

    assert torch.equal(restored.model.featurizer.card_encoder.id_emb.weight, weights)


def test_learned_ppo_value_head_uses_the_shared_embedding():
    """The critic consumes embedded state, so values stay finite and shaped."""
    agent = create_battle_agent("PPO", policy="learned")
    states = torch.tensor(
        [agent.encode_state(BATTLE_STATE)], dtype=torch.float32, device=agent.device
    )

    value = agent.model.value(states)

    assert value.shape == (1,)
    assert torch.isfinite(value).all()


def test_flat_variant_layout_is_unchanged():
    """The default path must stay byte-compatible with existing checkpoints."""
    agent = create_battle_agent("PPO")

    assert agent.ACTION_SCHEMA == "candidate_action_ppo_v3"
    assert agent.model_input_size == BattleStateEncoder().model_input_size

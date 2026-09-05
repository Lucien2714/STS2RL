"""Structured token and numeric conversion tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from sts2rl.encoder import (
    EntityReference,
    NumericFeature,
    TokenizedAction,
    TokenizedDecision,
    TokenizedEntityBatch,
    TokenizedMap,
    TokenizedState,
    linear_feature,
    pack_numeric,
    ratio_feature,
    signed_log_feature,
)
from sts2rl.env import GameObservation


def _empty_numeric() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.bool)


def _state_with_card() -> TokenizedState:
    numeric, numeric_mask = _empty_numeric()
    return TokenizedState(
        global_categorical=torch.tensor([2], dtype=torch.long),
        global_numeric=numeric,
        global_numeric_mask=numeric_mask,
        entities={
            "card": TokenizedEntityBatch(
                kind="card",
                categorical=torch.tensor([[5, 3]], dtype=torch.long),
                numeric=numeric.reshape(1, 0),
                numeric_mask=numeric_mask.reshape(1, 0),
            ),
        },
    )


def test_game_observation_is_frozen_without_changing_raw_dictionaries():
    raw_state = {"state_type": "map"}
    observation = GameObservation(raw_state)

    assert observation.raw_state is raw_state
    with pytest.raises(FrozenInstanceError):
        observation.raw_state = {}  # type: ignore[misc]


def test_numeric_features_distinguish_present_zero_from_missing_values():
    values, mask = pack_numeric(
        [
            linear_feature(0),
            linear_feature(None),
            signed_log_feature(-3),
            ratio_feature(5, 10),
            ratio_feature(5, 0),
        ]
    )

    assert values.dtype == torch.float32
    assert mask.dtype == torch.bool
    assert values[0].item() == 0.0
    assert mask.tolist() == [True, False, True, True, False]
    assert values[1].item() == 0.0
    assert values[2].item() == pytest.approx(-math.log1p(3))
    assert values[3].item() == pytest.approx(0.5)


@pytest.mark.parametrize("value", [True, None, "X", float("nan"), float("inf")])
def test_invalid_numeric_values_are_marked_missing(value):
    feature = signed_log_feature(value)

    assert feature.value == 0.0
    assert feature.present is False


def test_numeric_feature_constructors_reject_non_finite_present_values():
    with pytest.raises(ValueError, match="finite"):
        NumericFeature(float("inf"), True)
    with pytest.raises(ValueError, match="missing"):
        NumericFeature(1.0, False)


def test_tokenized_decision_moves_recursively_without_mutating_original():
    state = _state_with_card()
    numeric, numeric_mask = _empty_numeric()
    decision = TokenizedDecision(
        state=state,
        actions=(
            TokenizedAction(
                action_type=torch.tensor(2, dtype=torch.long),
                numeric=numeric,
                numeric_mask=numeric_mask,
                source=EntityReference("card", 0),
            ),
        ),
    )

    moved = decision.to("meta")

    assert decision.state.global_categorical.device.type == "cpu"
    assert decision.actions[0].action_type.device.type == "cpu"
    assert moved is not decision
    assert moved.state.global_categorical.device.type == "meta"
    assert moved.state.entities["card"].numeric.device.type == "meta"
    assert moved.actions[0].action_type.device.type == "meta"


def test_decision_rejects_an_entity_reference_with_an_unknown_kind():
    state = _state_with_card()
    numeric, numeric_mask = _empty_numeric()

    with pytest.raises(ValueError, match="unknown kind"):
        TokenizedDecision(
            state=state,
            actions=(
                TokenizedAction(
                    action_type=torch.tensor(2, dtype=torch.long),
                    numeric=numeric,
                    numeric_mask=numeric_mask,
                    source=EntityReference("relic", 0),
                ),
            ),
        )


def test_entity_batches_fill_empty_relationship_rows_and_validate_lengths():
    batch = TokenizedEntityBatch(
        kind="card",
        categorical=torch.tensor([[2], [3]], dtype=torch.long),
        numeric=torch.zeros((2, 1), dtype=torch.float32),
        numeric_mask=torch.ones((2, 1), dtype=torch.bool),
    )

    assert batch.entity_count == 2
    assert batch.active.tolist() == [False, False]
    assert batch.active_mask.tolist() == [False, False]
    assert batch.owners == (None, None)
    assert batch.children == ((), ())

    with pytest.raises(ValueError, match="one value per row"):
        TokenizedEntityBatch(
            kind="card",
            categorical=batch.categorical,
            numeric=batch.numeric,
            numeric_mask=batch.numeric_mask,
            owners=(None,),
        )


def test_entity_activity_distinguishes_active_inactive_and_unknown():
    batch = TokenizedEntityBatch(
        kind="relic",
        categorical=torch.tensor([[2], [3], [4]], dtype=torch.long),
        numeric=torch.zeros((3, 1), dtype=torch.float32),
        numeric_mask=torch.ones((3, 1), dtype=torch.bool),
        active=torch.tensor([True, False, False], dtype=torch.bool),
        active_mask=torch.tensor([True, True, False], dtype=torch.bool),
    )

    assert batch.active.tolist() == [True, False, False]
    assert batch.active_mask.tolist() == [True, True, False]

    with pytest.raises(ValueError, match="supplied together"):
        TokenizedEntityBatch(
            kind="relic",
            categorical=torch.tensor([[2]], dtype=torch.long),
            numeric=torch.zeros((1, 1), dtype=torch.float32),
            numeric_mask=torch.ones((1, 1), dtype=torch.bool),
            active=torch.tensor([True], dtype=torch.bool),
        )


def test_state_copies_entity_mapping_and_requires_matching_batch_kind():
    state = _state_with_card()

    with pytest.raises(TypeError):
        state.entities["relic"] = state.entities["card"]  # type: ignore[index]

    with pytest.raises(ValueError, match="key must match"):
        TokenizedState(
            global_categorical=state.global_categorical,
            global_numeric=state.global_numeric,
            global_numeric_mask=state.global_numeric_mask,
            entities={"relic": state.entities["card"]},
        )


def test_tokenized_map_validates_shapes_and_node_indices():
    game_map = TokenizedMap(
        node_categorical=torch.tensor([[2], [3], [4]], dtype=torch.long),
        node_numeric=torch.zeros((3, 2), dtype=torch.float32),
        node_numeric_mask=torch.ones((3, 2), dtype=torch.bool),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        reachable_mask=torch.tensor([True, True, True], dtype=torch.bool),
        candidate_indices=torch.tensor([1], dtype=torch.long),
        boss_indices=torch.tensor([2], dtype=torch.long),
        candidate_type_counts=torch.ones((1, 4), dtype=torch.float32),
        current_index=0,
    )

    assert game_map.edge_index.shape == (2, 2)

    with pytest.raises(ValueError, match="out-of-range node index"):
        TokenizedMap(
            node_categorical=game_map.node_categorical,
            node_numeric=game_map.node_numeric,
            node_numeric_mask=game_map.node_numeric_mask,
            edge_index=torch.tensor([[0, 1], [1, 9]], dtype=torch.long),
            reachable_mask=game_map.reachable_mask,
            candidate_indices=game_map.candidate_indices,
            boss_indices=game_map.boss_indices,
            candidate_type_counts=game_map.candidate_type_counts,
            current_index=game_map.current_index,
        )

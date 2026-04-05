import torch
import pytest
from app.signals.geometric import compute_all_geometric


def _make_mock_output(similar_layers=True):
    states = []
    for i in range(25):
        if similar_layers:
            state = torch.ones(1, 10, 896) * (1.0 + i * 0.01)
        else:
            state = torch.randn(1, 10, 896) * (1.0 + i * 0.5)
        states.append(state)
    return {"hidden_states": tuple(states), "logits": torch.randn(1, 10, 32000), "input_ids": torch.randint(0, 32000, (1, 10))}


def test_similar_layers_short_trajectory():
    output = _make_mock_output(similar_layers=True)
    result = compute_all_geometric(output)
    assert result["embedding_trajectory_length"]["value"] >= 0


def test_divergent_layers_high_trajectory():
    similar = _make_mock_output(similar_layers=True)
    divergent = _make_mock_output(similar_layers=False)
    r_similar = compute_all_geometric(similar)
    r_divergent = compute_all_geometric(divergent)
    assert r_divergent["embedding_trajectory_length"]["value"] > r_similar["embedding_trajectory_length"]["value"]


def test_returns_both_signals():
    output = _make_mock_output()
    result = compute_all_geometric(output)
    assert "embedding_trajectory_length" in result
    assert "layerwise_cosine_similarity" in result


def test_none_input():
    result = compute_all_geometric(None)
    assert result["embedding_trajectory_length"]["value"] == 0.0

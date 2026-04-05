import torch
import numpy as np
import pytest
from app.signals.layer_wise import compute_all_layer_wise


def _make_hidden_states(num_layers: int, seq_len: int, hidden_dim: int, confident: bool):
    states = []
    for i in range(num_layers + 1):
        if confident:
            state = torch.randn(1, seq_len, hidden_dim) * 0.1 + i * 0.01
        else:
            state = torch.randn(1, seq_len, hidden_dim) * (1.0 + i * 0.5)
        states.append(state)
    return tuple(states)


def _make_mock_output(num_layers=24, seq_len=10, hidden_dim=896, confident=True):
    hidden_states = _make_hidden_states(num_layers, seq_len, hidden_dim, confident)
    logits = torch.randn(1, seq_len, 32000)
    input_ids = torch.randint(0, 32000, (1, seq_len))
    return {"hidden_states": hidden_states, "logits": logits, "input_ids": input_ids}


def test_confident_output_has_low_icr():
    output = _make_mock_output(confident=True)
    result = compute_all_layer_wise(output)
    assert result["icr_score"]["value"] >= 0.0
    assert result["icr_score"]["value"] <= 1.0


def test_returns_all_four_signals():
    output = _make_mock_output()
    result = compute_all_layer_wise(output)
    assert "dola_contrast" in result
    assert "logit_lens_evolution" in result
    assert "icr_score" in result
    assert "prediction_stability" in result


def test_none_input_returns_defaults():
    result = compute_all_layer_wise(None)
    assert result["dola_contrast"]["value"] == 0.0
    assert result["icr_score"]["value"] == 0.0
    assert result["prediction_stability"]["value"] == 0

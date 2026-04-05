import pytest
from app.signals.calibration import compute_confidence_agreement, compute_composite_score


def test_confidence_agreement_match():
    result = compute_confidence_agreement(
        verbalized_confidence=0.9,
        signal_confidence=0.85,
    )
    assert result["value"] > 0.8


def test_confidence_agreement_mismatch():
    result = compute_confidence_agreement(
        verbalized_confidence=0.95,
        signal_confidence=0.2,
    )
    assert result["value"] < 0.5


def test_composite_score_all_confident():
    signals = {
        "predictive_entropy": {"value": 0.2},
        "mean_token_prob": {"value": 0.9},
        "self_consistency": {"value": 1.0},
        "hedging_score": {"value": 0.0},
        "icr_score": {"value": 0.05},
    }
    result = compute_composite_score(signals)
    assert result["value"] > 70


def test_composite_score_all_uncertain():
    signals = {
        "predictive_entropy": {"value": 3.0},
        "mean_token_prob": {"value": 0.1},
        "self_consistency": {"value": 0.2},
        "hedging_score": {"value": 0.8},
        "icr_score": {"value": 0.8},
    }
    result = compute_composite_score(signals)
    assert result["value"] < 40


def test_composite_score_empty():
    result = compute_composite_score({})
    assert result["value"] == 50

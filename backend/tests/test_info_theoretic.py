import numpy as np
import pytest
from app.signals.info_theoretic import compute_all_info_theoretic


def test_high_confidence_signals():
    """When logprobs are high (near 0), all signals should indicate confidence."""
    logprobs = [np.log(0.95)] * 20
    result = compute_all_info_theoretic(logprobs, top_logprobs=None)

    assert result["predictive_entropy"]["value"] < 0.5
    assert result["perplexity"]["value"] < 2.0
    assert result["mean_token_prob"]["value"] > 0.9
    assert result["min_token_prob"]["value"] > 0.9
    assert result["token_prob_variance"]["value"] < 0.01


def test_low_confidence_signals():
    """When logprobs are low, all signals should indicate uncertainty."""
    logprobs = [np.log(0.1)] * 20
    result = compute_all_info_theoretic(logprobs, top_logprobs=None)

    assert result["predictive_entropy"]["value"] > 1.0
    assert result["perplexity"]["value"] > 5.0
    assert result["mean_token_prob"]["value"] < 0.2
    assert result["min_token_prob"]["value"] < 0.2


def test_mixed_confidence():
    """Mixed logprobs should show high variance."""
    logprobs = [np.log(0.95)] * 10 + [np.log(0.1)] * 10
    result = compute_all_info_theoretic(logprobs, top_logprobs=None)

    assert result["token_prob_variance"]["value"] > 0.05


def test_top_k_with_top_logprobs():
    """Top-k mass should reflect concentration in top tokens."""
    top_logprobs = [
        [{"logprob": np.log(0.9)}, {"logprob": np.log(0.05)}, {"logprob": np.log(0.03)}]
    ] * 10
    logprobs = [np.log(0.9)] * 10
    result = compute_all_info_theoretic(logprobs, top_logprobs=top_logprobs)

    assert result["top_k_prob_mass"]["value"] > 0.9


def test_empty_logprobs_returns_defaults():
    """Empty logprobs should return safe defaults."""
    result = compute_all_info_theoretic([], top_logprobs=None)

    assert result["predictive_entropy"]["value"] == 0.0
    assert result["perplexity"]["value"] == 1.0
    assert result["mean_token_prob"]["value"] == 0.0

import pytest
from app.signals.behavioral import (
    compute_hedging_score,
    compute_self_consistency,
    compute_response_length_variance,
    compute_semantic_similarity,
)


def test_hedging_high():
    text = "I'm not sure, but I think it might be approximately 42, perhaps."
    result = compute_hedging_score(text)
    assert result["value"] > 0.3


def test_hedging_low():
    text = "The answer is 42. This is a well-established fact."
    result = compute_hedging_score(text)
    assert result["value"] < 0.2


def test_self_consistency_identical():
    responses = ["Paris"] * 5
    result = compute_self_consistency(responses)
    assert result["value"] == 1.0


def test_self_consistency_mixed():
    responses = ["Paris", "Paris", "London", "Berlin", "Paris"]
    result = compute_self_consistency(responses)
    assert result["value"] == 0.6


def test_self_consistency_empty():
    result = compute_self_consistency([])
    assert result["value"] == 0.0


def test_response_length_variance_same():
    responses = ["Hello world", "Hi there!!", "Hey planet"]
    result = compute_response_length_variance(responses)
    assert result["value"] < 100


def test_response_length_variance_different():
    responses = ["Hi", "This is a much longer response with many more words and details"]
    result = compute_response_length_variance(responses)
    assert result["value"] > 100


def test_semantic_similarity_identical():
    responses = ["Paris is the capital"] * 3
    result = compute_semantic_similarity(responses)
    assert result["value"] == 1.0

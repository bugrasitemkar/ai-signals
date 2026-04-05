import pytest
from app.signals.engine import compute_fast_signals


def test_fast_signals_with_logprobs():
    import numpy as np
    logprobs = [np.log(0.9)] * 10
    result = compute_fast_signals(logprobs, top_logprobs=None, response_text="Paris is the capital.")
    assert "predictive_entropy" in result
    assert "perplexity" in result
    assert "hedging_score" in result
    assert len(result) == 7


def test_fast_signals_empty():
    result = compute_fast_signals([], top_logprobs=None, response_text="")
    assert "predictive_entropy" in result

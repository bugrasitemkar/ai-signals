import numpy as np


def compute_all_info_theoretic(
    logprobs: list[float],
    top_logprobs: list[list[dict]] | None,
) -> dict:
    """Compute all 6 information-theoretic signals from token logprobs."""
    if not logprobs:
        return {
            "predictive_entropy": _signal("predictive_entropy", 0.0, "No tokens to analyze"),
            "perplexity": _signal("perplexity", 1.0, "No tokens to analyze"),
            "mean_token_prob": _signal("mean_token_prob", 0.0, "No tokens to analyze"),
            "min_token_prob": _signal("min_token_prob", 0.0, "No tokens to analyze"),
            "top_k_prob_mass": _signal("top_k_prob_mass", 0.0, "No tokens to analyze"),
            "token_prob_variance": _signal("token_prob_variance", 0.0, "No tokens to analyze"),
        }

    lp = np.array(logprobs, dtype=np.float64)
    probs = np.exp(lp)

    entropy = _predictive_entropy(probs, lp)
    perplexity = _perplexity(lp)
    mean_tp = float(np.mean(probs))
    min_tp = float(np.min(probs))
    variance = float(np.var(probs))
    top_k = _top_k_mass(top_logprobs)

    return {
        "predictive_entropy": _signal(
            "predictive_entropy",
            entropy,
            _interpret_entropy(entropy),
        ),
        "perplexity": _signal(
            "perplexity",
            perplexity,
            _interpret_perplexity(perplexity),
        ),
        "mean_token_prob": _signal(
            "mean_token_prob",
            mean_tp,
            _interpret_mean_tp(mean_tp),
        ),
        "min_token_prob": _signal(
            "min_token_prob",
            min_tp,
            _interpret_min_tp(min_tp),
        ),
        "top_k_prob_mass": _signal(
            "top_k_prob_mass",
            top_k,
            _interpret_top_k(top_k),
        ),
        "token_prob_variance": _signal(
            "token_prob_variance",
            variance,
            _interpret_variance(variance),
        ),
    }


def _predictive_entropy(probs: np.ndarray, logprobs: np.ndarray) -> float:
    """H(Y|x) = -Σ p(y|x) log p(y|x)"""
    entropy = -np.sum(probs * logprobs)
    return float(max(0.0, entropy))


def _perplexity(logprobs: np.ndarray) -> float:
    """PPL = exp(-(1/n) Σ log p(tᵢ))"""
    return float(np.exp(-np.mean(logprobs)))


def _top_k_mass(top_logprobs: list[list[dict]] | None) -> float:
    """Average probability mass in top-k tokens across all positions."""
    if not top_logprobs:
        return 0.0

    masses = []
    for position in top_logprobs:
        if position:
            position_mass = sum(np.exp(t["logprob"]) for t in position)
            masses.append(position_mass)

    return float(np.mean(masses)) if masses else 0.0


def _signal(signal_id: str, value: float, interpretation: str) -> dict:
    return {
        "signal_id": signal_id,
        "school": "information_theoretic",
        "value": round(value, 4),
        "interpretation": interpretation,
    }


def _interpret_entropy(v: float) -> str:
    if v < 0.5:
        return "Low entropy — model is confident in this response"
    if v < 2.0:
        return "Moderate entropy — some uncertainty in token choices"
    return "High entropy — model is uncertain, probability spread across many tokens"


def _interpret_perplexity(v: float) -> str:
    if v < 2.0:
        return "Model is minimally surprised by its own output"
    if v < 10.0:
        return "Moderate perplexity — multiple plausible tokens at some positions"
    return "High perplexity — model found its own output quite surprising"


def _interpret_mean_tp(v: float) -> str:
    if v > 0.8:
        return "High average confidence across all tokens"
    if v > 0.5:
        return "Moderate average confidence — some tokens chosen with less certainty"
    return "Low average confidence — many tokens were not the model's top choice"


def _interpret_min_tp(v: float) -> str:
    if v > 0.5:
        return "Even the least confident token had reasonable probability"
    if v > 0.1:
        return "At least one token was chosen with low confidence"
    return "Contains a token the model was very unsure about — potential weak point"


def _interpret_top_k(v: float) -> str:
    if v > 0.9:
        return "Probability heavily concentrated in top tokens"
    if v > 0.7:
        return "Most probability in top tokens, some spread"
    return "Probability spread thin across many tokens — low concentration"


def _interpret_variance(v: float) -> str:
    if v < 0.01:
        return "Very consistent confidence across all tokens"
    if v < 0.05:
        return "Some variation in confidence across the response"
    return "High variance — model confidence fluctuates significantly across tokens"

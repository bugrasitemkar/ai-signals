import numpy as np


def compute_confidence_agreement(
    verbalized_confidence: float,
    signal_confidence: float,
) -> dict:
    """Compare model's stated confidence with signal-derived confidence."""
    agreement = 1.0 - abs(verbalized_confidence - signal_confidence)
    agreement = round(max(0.0, min(1.0, agreement)), 4)
    return _signal(
        "confidence_agreement",
        agreement,
        _interpret_agreement(agreement),
        metadata={
            "verbalized": round(verbalized_confidence, 4),
            "signal_derived": round(signal_confidence, 4),
        },
    )


def compute_composite_score(signals: dict) -> dict:
    """Compute weighted aggregate reliability score (0-100)."""
    if not signals:
        return _signal("composite_score", 50, "No signals available for scoring")

    scores = []
    weights = []

    if "predictive_entropy" in signals:
        entropy = signals["predictive_entropy"]["value"]
        score = max(0, min(100, 100 - (entropy / 3.0) * 100))
        scores.append(score)
        weights.append(0.2)

    if "mean_token_prob" in signals:
        mtp = signals["mean_token_prob"]["value"]
        scores.append(mtp * 100)
        weights.append(0.15)

    if "self_consistency" in signals:
        sc = signals["self_consistency"]["value"]
        scores.append(sc * 100)
        weights.append(0.25)

    if "hedging_score" in signals:
        hedge = signals["hedging_score"]["value"]
        scores.append((1 - hedge) * 100)
        weights.append(0.1)

    if "icr_score" in signals:
        icr = signals["icr_score"]["value"]
        scores.append((1 - icr) * 100)
        weights.append(0.15)

    if not scores:
        return _signal("composite_score", 50, "No scorable signals available")

    total_weight = sum(weights)
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    composite = round(max(0, min(100, weighted_score)), 1)
    return _signal("composite_score", composite, _interpret_composite(composite))


def _signal(signal_id: str, value, interpretation: str, metadata: dict = None) -> dict:
    result = {
        "signal_id": signal_id,
        "school": "calibration",
        "value": value,
        "interpretation": interpretation,
    }
    if metadata:
        result["metadata"] = metadata
    return result


def _interpret_agreement(v: float) -> str:
    if v > 0.8:
        return "Good agreement — stated confidence aligns with signal evidence"
    if v > 0.5:
        return "Moderate agreement — some gap between stated and measured confidence"
    return "Poor agreement — model's stated confidence does not match its signals"


def _interpret_composite(v: float) -> str:
    if v >= 80:
        return "High reliability — multiple signals converge on confidence"
    if v >= 50:
        return "Moderate reliability — mixed signals, verify key claims"
    return "Low reliability — signals indicate significant uncertainty"

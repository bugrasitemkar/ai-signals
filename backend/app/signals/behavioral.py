import re
import numpy as np
from collections import Counter

HEDGING_PATTERNS = [
    r"\bi'm not (sure|certain)\b",
    r"\bit's possible\b",
    r"\bperhaps\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bi think\b",
    r"\bapproximately\b",
    r"\broughly\b",
    r"\baround\b",
    r"\bmaybe\b",
]


def compute_hedging_score(text: str) -> dict:
    """Detect hedging language indicating uncertainty."""
    text_lower = text.lower()
    matches = sum(1 for p in HEDGING_PATTERNS if re.search(p, text_lower))
    score = round(matches / len(HEDGING_PATTERNS), 4)
    return _signal("hedging_score", score, _interpret_hedging(score))


def compute_self_consistency(responses: list[str]) -> dict:
    """Compute agreement across multiple sampled responses."""
    if not responses:
        return _signal("self_consistency", 0.0, "No responses to compare")
    normalized = [r.strip().lower() for r in responses]
    counts = Counter(normalized)
    majority_count = counts.most_common(1)[0][1]
    score = round(majority_count / len(responses), 4)
    return _signal(
        "self_consistency",
        score,
        _interpret_consistency(score),
        metadata={
            "num_samples": len(responses),
            "num_distinct": len(counts),
            "majority_answer": counts.most_common(1)[0][0][:100],
        },
    )


def compute_response_length_variance(responses: list[str]) -> dict:
    """Compute variance in response lengths across samples."""
    if len(responses) < 2:
        return _signal("response_length_variance", 0.0, "Need multiple responses")
    lengths = [len(r) for r in responses]
    variance = round(float(np.var(lengths)), 4)
    return _signal("response_length_variance", variance, _interpret_length_var(variance))


def compute_semantic_similarity(responses: list[str]) -> dict:
    """Compute average pairwise similarity using character n-gram overlap."""
    if len(responses) < 2:
        return _signal("semantic_similarity", 0.0, "Need multiple responses")

    def ngram_set(text: str, n: int = 3) -> set:
        text = text.lower().strip()
        return {text[i : i + n] for i in range(len(text) - n + 1)} if len(text) >= n else {text}

    sims = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            set_a = ngram_set(responses[i])
            set_b = ngram_set(responses[j])
            if set_a or set_b:
                jaccard = len(set_a & set_b) / len(set_a | set_b)
                sims.append(jaccard)

    score = round(float(np.mean(sims)), 4) if sims else 0.0
    return _signal("semantic_similarity", score, _interpret_semantic_sim(score))


def _signal(signal_id: str, value, interpretation: str, metadata: dict = None) -> dict:
    result = {
        "signal_id": signal_id,
        "school": "behavioral",
        "value": value,
        "interpretation": interpretation,
    }
    if metadata:
        result["metadata"] = metadata
    return result


def _interpret_hedging(v: float) -> str:
    if v < 0.1:
        return "No hedging language detected — model states claims directly"
    if v < 0.3:
        return "Some hedging language — model expresses mild caution"
    return "Significant hedging — model uses uncertain language throughout"


def _interpret_consistency(v: float) -> str:
    if v > 0.8:
        return "High consistency — model gives the same answer across samples"
    if v > 0.5:
        return "Moderate consistency — majority agreement but some variation"
    return "Low consistency — model gives different answers each time"


def _interpret_length_var(v: float) -> str:
    if v < 100:
        return "Consistent response lengths — stable output structure"
    if v < 1000:
        return "Some variation in response length across samples"
    return "High length variance — model produces very different response sizes"


def _interpret_semantic_sim(v: float) -> str:
    if v > 0.8:
        return "High semantic similarity — responses convey consistent meaning"
    if v > 0.5:
        return "Moderate similarity — responses share core content but differ in detail"
    return "Low similarity — responses express substantially different content"

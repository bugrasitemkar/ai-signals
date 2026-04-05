import asyncio
from app.signals.info_theoretic import compute_all_info_theoretic
from app.signals.layer_wise import compute_all_layer_wise
from app.signals.geometric import compute_all_geometric
from app.signals.behavioral import (
    compute_hedging_score,
    compute_self_consistency,
    compute_response_length_variance,
    compute_semantic_similarity,
)
from app.signals.calibration import compute_composite_score
from app.models.hf_model import extract_hidden_states
from app.models.ollama_client import sample_responses


def compute_fast_signals(
    logprobs: list[float],
    top_logprobs: list[list[dict]] | None,
    response_text: str,
) -> dict:
    """Compute instant signals (info-theoretic + hedging). Returns in <0.5s."""
    info_signals = compute_all_info_theoretic(logprobs, top_logprobs)
    hedging = compute_hedging_score(response_text)
    return {**info_signals, hedging["signal_id"]: hedging}


def compute_layer_signals(question: str) -> dict:
    """Compute layer-wise and geometric signals via HuggingFace. Takes 3-5s."""
    model_output = extract_hidden_states(question)
    layer_signals = compute_all_layer_wise(model_output)
    geo_signals = compute_all_geometric(model_output)
    return {**layer_signals, **geo_signals}


async def compute_behavioral_signals(question: str) -> dict:
    """Compute behavioral signals via Ollama multi-sampling. Takes 5-8s."""
    responses = await sample_responses(question, n=5)
    consistency = compute_self_consistency(responses)
    length_var = compute_response_length_variance(responses)
    semantic_sim = compute_semantic_similarity(responses)
    return {
        consistency["signal_id"]: consistency,
        length_var["signal_id"]: length_var,
        semantic_sim["signal_id"]: semantic_sim,
    }

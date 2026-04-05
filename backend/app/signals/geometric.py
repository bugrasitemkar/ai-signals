import torch
import torch.nn.functional as F
import numpy as np


def compute_all_geometric(model_output: dict | None) -> dict:
    """Compute 2 geometric signals from hidden states."""
    if model_output is None:
        return {
            "embedding_trajectory_length": _signal("embedding_trajectory_length", 0.0, "HuggingFace model not available"),
            "layerwise_cosine_similarity": _signal("layerwise_cosine_similarity", 0.0, "HuggingFace model not available"),
        }

    hidden_states = model_output["hidden_states"]
    trajectory = _trajectory_length(hidden_states)
    cosine_sim = _layerwise_cosine_sim(hidden_states)

    return {
        "embedding_trajectory_length": _signal(
            "embedding_trajectory_length", trajectory, _interpret_trajectory(trajectory)
        ),
        "layerwise_cosine_similarity": _signal(
            "layerwise_cosine_similarity", cosine_sim, _interpret_cosine(cosine_sim)
        ),
    }


def _trajectory_length(hidden_states: tuple) -> float:
    total = 0.0
    for i in range(1, len(hidden_states)):
        prev = hidden_states[i - 1][:, -1, :]
        curr = hidden_states[i][:, -1, :]
        dist = torch.norm(curr - prev, dim=-1).item()
        total += dist
    return round(total, 4)


def _layerwise_cosine_sim(hidden_states: tuple) -> float:
    sims = []
    for i in range(1, len(hidden_states)):
        prev = hidden_states[i - 1][:, -1, :]
        curr = hidden_states[i][:, -1, :]
        sim = F.cosine_similarity(prev, curr, dim=-1).item()
        sims.append(sim)
    return round(float(np.mean(sims)), 4) if sims else 0.0


def _signal(signal_id: str, value, interpretation: str) -> dict:
    return {
        "signal_id": signal_id,
        "school": "geometric",
        "value": value,
        "interpretation": interpretation,
    }


def _interpret_trajectory(v: float) -> str:
    if v < 10.0:
        return "Short trajectory — representations change little across layers"
    if v < 50.0:
        return "Moderate trajectory — typical transformation through layers"
    return "Long trajectory — significant representational changes, complex processing"


def _interpret_cosine(v: float) -> str:
    if v > 0.95:
        return "Very high layer similarity — minimal transformation between layers"
    if v > 0.8:
        return "High similarity — gradual, smooth changes across layers"
    return "Lower similarity — significant transformations between layers"

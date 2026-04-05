import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon


def compute_all_layer_wise(model_output: dict | None) -> dict:
    """Compute all 4 layer-wise signals from HuggingFace model output."""
    if model_output is None:
        return {
            "dola_contrast": _signal("dola_contrast", 0.0, "HuggingFace model not available"),
            "logit_lens_evolution": _signal("logit_lens_evolution", [], "HuggingFace model not available"),
            "icr_score": _signal("icr_score", 0.0, "HuggingFace model not available"),
            "prediction_stability": _signal("prediction_stability", 0, "HuggingFace model not available"),
        }

    hidden_states = model_output["hidden_states"]
    logits = model_output["logits"]
    num_layers = len(hidden_states) - 1

    dola = _dola_contrast(hidden_states, logits)
    lens_evolution = _logit_lens_evolution(hidden_states, logits)
    icr = _icr_score(hidden_states, logits)
    stability = _prediction_stability(lens_evolution)

    return {
        "dola_contrast": _signal("dola_contrast", dola, _interpret_dola(dola)),
        "logit_lens_evolution": _signal(
            "logit_lens_evolution", lens_evolution, _interpret_lens(lens_evolution)
        ),
        "icr_score": _signal("icr_score", icr, _interpret_icr(icr)),
        "prediction_stability": _signal(
            "prediction_stability", stability, _interpret_stability(stability, num_layers)
        ),
    }


def _dola_contrast(hidden_states: tuple, logits: torch.Tensor) -> float:
    num_layers = len(hidden_states) - 1
    early_idx = max(1, num_layers // 4)
    early_hidden = hidden_states[early_idx][:, -1, :]
    final_hidden = hidden_states[-1][:, -1, :]
    cos_sim = F.cosine_similarity(early_hidden, final_hidden, dim=-1)
    contrast = 1.0 - cos_sim.item()
    return round(float(contrast), 4)


def _logit_lens_evolution(hidden_states: tuple, logits: torch.Tensor) -> list:
    evolution = []
    num_layers = len(hidden_states) - 1
    sample_indices = np.linspace(1, num_layers, min(8, num_layers), dtype=int)
    for idx in sample_indices:
        hidden = hidden_states[idx][:, -1, :]
        norm = torch.norm(hidden, dim=-1).item()
        evolution.append({"layer": int(idx), "norm": round(norm, 4)})
    return evolution


def _icr_score(hidden_states: tuple, logits: torch.Tensor) -> float:
    num_layers = len(hidden_states) - 1
    early_idx = max(1, num_layers // 4)
    early_hidden = hidden_states[early_idx][:, -1, :].squeeze()
    final_hidden = hidden_states[-1][:, -1, :].squeeze()
    early_dist = F.softmax(early_hidden, dim=-1).numpy()
    final_dist = F.softmax(final_hidden, dim=-1).numpy()
    jsd = jensenshannon(early_dist, final_dist)
    return round(float(jsd), 4) if not np.isnan(jsd) else 0.0


def _prediction_stability(lens_evolution: list) -> int:
    if len(lens_evolution) < 2:
        return 0
    norms = [e["norm"] for e in lens_evolution]
    changes = 0
    for i in range(1, len(norms)):
        relative_change = abs(norms[i] - norms[i - 1]) / (norms[i - 1] + 1e-10)
        if relative_change > 0.1:
            changes += 1
    return changes


def _signal(signal_id: str, value, interpretation: str) -> dict:
    return {
        "signal_id": signal_id,
        "school": "layer_wise",
        "value": value,
        "interpretation": interpretation,
    }


def _interpret_dola(v: float) -> str:
    if v < 0.1:
        return "Low contrast — early and late layers agree, suggesting factual grounding"
    if v < 0.3:
        return "Moderate contrast — some difference between layer predictions"
    return "High contrast — significant disagreement between early and late layers"


def _interpret_lens(evolution: list) -> str:
    if not evolution:
        return "No layer data available"
    norms = [e["norm"] for e in evolution]
    trend = norms[-1] - norms[0]
    if abs(trend) < 1.0:
        return "Stable predictions across layers — consistent reasoning"
    if trend > 0:
        return "Increasing activation through layers — model building confidence"
    return "Decreasing activation through layers — unusual pattern"


def _interpret_icr(v: float) -> str:
    if v < 0.2:
        return "Low ICR — early and late layers agree, model did not change its mind"
    if v < 0.5:
        return "Moderate ICR — some internal disagreement across layers"
    return "High ICR — model's early and late layers diverge significantly"


def _interpret_stability(changes: int, num_layers: int) -> str:
    ratio = changes / max(1, num_layers)
    if ratio < 0.1:
        return "Highly stable — predictions barely change across layers"
    if ratio < 0.3:
        return "Moderately stable — some prediction shifts through layers"
    return "Unstable — frequent prediction changes, suggesting uncertainty"

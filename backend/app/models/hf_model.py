"""
Loads Qwen2.5-0.5B on CPU for hidden state extraction.
The model is loaded once at startup and reused for all requests.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-0.5B")

_tokenizer = None
_model = None


def is_hf_model_loaded() -> bool:
    """Check if the model is already loaded without triggering a load."""
    return _model is not None


def get_hf_model():
    """Get or lazily load the HuggingFace model."""
    global _tokenizer, _model
    if _model is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            _model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                torch_dtype=torch.float32,
            )
            _model.eval()
        except Exception as e:
            print(f"[hf_model] Failed to load {HF_MODEL_NAME}: {e}")
            return None
    return _model


def get_hf_tokenizer():
    """Get or lazily load the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        get_hf_model()
    return _tokenizer


def extract_hidden_states(text: str) -> dict | None:
    """
    Run text through HF model and extract hidden states at all layers.

    Returns:
        Dict with 'hidden_states' (list of tensors, one per layer),
        'logits' (final layer logits), and 'input_ids'.
        Returns None if model is not loaded.
    """
    model = get_hf_model()
    tokenizer = get_hf_tokenizer()

    if model is None or tokenizer is None:
        return None

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return {
        "hidden_states": outputs.hidden_states,
        "logits": outputs.logits,
        "input_ids": inputs["input_ids"],
    }

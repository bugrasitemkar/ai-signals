"""Download HuggingFace model on first startup. Pulls Ollama model if not present."""
import os
import time
import httpx

HF_MODEL = "Qwen/Qwen2.5-0.5B"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:1b")


def download_hf_model():
    """Pre-download HuggingFace model so first query isn't slow."""
    print(f"[setup] Checking HuggingFace model: {HF_MODEL}")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        AutoTokenizer.from_pretrained(HF_MODEL)
        AutoModelForCausalLM.from_pretrained(HF_MODEL)
        print(f"[setup] HuggingFace model ready: {HF_MODEL}")
    except Exception as e:
        print(f"[setup] HuggingFace model download failed: {e}")
        print("[setup] Layer-wise and geometric signals will be unavailable.")


def wait_for_ollama(max_wait: int = 60):
    """Wait for Ollama to be reachable."""
    for i in range(max_wait):
        try:
            httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return True
        except Exception:
            time.sleep(1)
    return False


def pull_ollama_model():
    """Pull default Ollama model if not already present."""
    print(f"[setup] Waiting for Ollama...")
    if not wait_for_ollama():
        print("[setup] Ollama not reachable, skipping model pull")
        return

    print(f"[setup] Checking Ollama model: {DEFAULT_MODEL}")
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        models = [m["name"] for m in response.json().get("models", [])]
        if DEFAULT_MODEL not in models and f"{DEFAULT_MODEL}:latest" not in models:
            print(f"[setup] Pulling {DEFAULT_MODEL}... (this may take a few minutes)")
            # Stream the pull response to avoid timeout
            with httpx.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": DEFAULT_MODEL},
                timeout=None,
            ) as r:
                for line in r.iter_lines():
                    if '"completed"' in line and '"total"' in line:
                        pass  # Progress update, ignore
            print(f"[setup] Ollama model ready: {DEFAULT_MODEL}")
        else:
            print(f"[setup] Ollama model already present: {DEFAULT_MODEL}")
    except Exception as e:
        print(f"[setup] Ollama model pull failed: {e}")
        print("[setup] You can pull manually: ollama pull " + DEFAULT_MODEL)


if __name__ == "__main__":
    pull_ollama_model()
    download_hf_model()

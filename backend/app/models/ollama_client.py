import os
import time
import uuid
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_active_model = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:1b")


def get_active_model() -> str:
    return _active_model


def set_active_model(model: str):
    global _active_model
    _active_model = model


async def chat(question: str, model: str | None = None) -> dict:
    """Send a chat request to Ollama with logprobs enabled via OpenAI-compatible API."""
    model = model or _active_model
    request_id = str(uuid.uuid4())
    start_time = time.time()

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Format your responses using markdown. Use headers, bold, lists, and code blocks where appropriate."},
                    {"role": "user", "content": question},
                ],
                "logprobs": True,
                "top_logprobs": 5,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        data = response.json()

    generation_time_ms = int((time.time() - start_time) * 1000)
    choice = data["choices"][0]
    content = choice["message"]["content"]

    logprobs_data = choice.get("logprobs", {}).get("content", [])
    token_logprobs = [t["logprob"] for t in logprobs_data] if logprobs_data else []
    top_logprobs = [t.get("top_logprobs", []) for t in logprobs_data] if logprobs_data else []

    return {
        "response": content,
        "request_id": request_id,
        "model": model,
        "generation_time_ms": generation_time_ms,
        "logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
    }


async def sample_responses(question: str, n: int = 5, model: str | None = None) -> list[str]:
    """Generate multiple responses for self-consistency analysis."""
    model = model or _active_model
    responses = []

    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(n):
            response = await client.post(
                f"{OLLAMA_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": question}],
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            data = response.json()
            responses.append(data["choices"][0]["message"]["content"])

    return responses


async def list_models() -> list[str]:
    """List available Ollama models."""
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]


async def generate_summary(prompt: str, model: str | None = None) -> str:
    """Generate text using Ollama (for executive summary)."""
    model = model or _active_model

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

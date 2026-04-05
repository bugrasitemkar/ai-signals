import os
import json
import httpx
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:1b")


@router.get("/api/health")
async def health_check():
    ollama_status = {"status": "disconnected", "model": None, "ready": False}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            # Match flexibly: "llama3.2:1b" matches "llama3.2:1b", "llama3.2:1b:latest", etc.
            model_base = DEFAULT_MODEL.split(":")[0]  # e.g. "llama3.2"
            has_model = len(models) > 0 and any(
                m.startswith(model_base) for m in models
            )
            ollama_status = {
                "status": "connected" if has_model else "downloading",
                "model": DEFAULT_MODEL if has_model else None,
                "ready": has_model,
            }
    except Exception:
        pass

    hf_status = {"status": "downloading", "model": None, "ready": False}
    try:
        from app.models.hf_model import is_hf_model_loaded
        if is_hf_model_loaded():
            hf_status = {"status": "loaded", "model": "Qwen/Qwen2.5-0.5B", "ready": True}
    except Exception:
        pass

    return {
        "ollama": ollama_status,
        "huggingface": hf_status,
        "ready": ollama_status["ready"],
    }


@router.get("/api/setup/stream")
async def setup_stream():
    """Stream model download progress to the frontend."""

    async def event_generator():
        # Check if model already exists
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if any(DEFAULT_MODEL in m for m in models):
                    yield {"event": "status", "data": json.dumps({
                        "stage": "ollama", "status": "ready", "progress": 100
                    })}
                    yield {"event": "done", "data": json.dumps({"status": "ready"})}
                    return
        except Exception:
            pass

        # Stream the pull progress
        yield {"event": "status", "data": json.dumps({
            "stage": "ollama", "status": "downloading", "model": DEFAULT_MODEL, "progress": 0
        })}

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/pull",
                    json={"name": DEFAULT_MODEL},
                ) as response:
                    async for line in response.aiter_lines():
                        try:
                            data = json.loads(line)
                            completed = data.get("completed", 0)
                            total = data.get("total", 0)
                            progress = int((completed / total) * 100) if total > 0 else 0
                            status_text = data.get("status", "downloading")
                            yield {"event": "status", "data": json.dumps({
                                "stage": "ollama",
                                "status": status_text,
                                "progress": progress,
                                "detail": f"{completed // (1024*1024)}MB / {total // (1024*1024)}MB" if total > 0 else status_text,
                            })}
                        except (json.JSONDecodeError, KeyError):
                            pass
        except Exception as e:
            yield {"event": "status", "data": json.dumps({
                "stage": "ollama", "status": "error", "detail": str(e)
            })}

        yield {"event": "status", "data": json.dumps({
            "stage": "ollama", "status": "ready", "progress": 100
        })}
        yield {"event": "done", "data": json.dumps({"status": "ready"})}

    return EventSourceResponse(event_generator())

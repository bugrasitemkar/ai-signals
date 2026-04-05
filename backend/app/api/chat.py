import os
import httpx
from fastapi import APIRouter, HTTPException

from app.schemas import ChatRequest, ChatResponse, ModelListResponse, ModelSwitchRequest
from app.models import ollama_client
from app.api.signals import store_request_data

router = APIRouter()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:1b")


async def _check_model_ready(model: str) -> bool:
    """Check if the requested model is available in Ollama."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            model_base = model.split(":")[0]
            return any(m.startswith(model_base) for m in models)
    except Exception:
        pass
    return False


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    model = request.model or DEFAULT_MODEL

    if not await _check_model_ready(model):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' is still downloading. Please wait a few minutes and try again.",
        )

    try:
        result = await ollama_client.chat(request.question, model)

        store_request_data(result["request_id"], {
            "question": request.question,
            "response": result["response"],
            "logprobs": result.get("logprobs", []),
            "top_logprobs": result.get("top_logprobs"),
        })

        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")


@router.get("/api/models", response_model=ModelListResponse)
async def list_models():
    try:
        models = await ollama_client.list_models()
        return ModelListResponse(
            models=models,
            active=ollama_client.get_active_model(),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")


@router.post("/api/models/switch")
async def switch_model(request: ModelSwitchRequest):
    models = await ollama_client.list_models()
    if request.model not in models:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    ollama_client.set_active_model(request.model)
    return {"active": request.model}

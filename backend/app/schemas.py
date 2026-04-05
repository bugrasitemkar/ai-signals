from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    model: str = "llama3.2:1b"


class ChatResponse(BaseModel):
    response: str
    request_id: str
    model: str
    generation_time_ms: int
    logprobs: list[float] | None = None
    top_logprobs: list[list[dict]] | None = None


class SignalResult(BaseModel):
    signal_id: str
    school: str
    value: float | list[float] | dict
    interpretation: str
    metadata: dict = {}


class SummaryResult(BaseModel):
    type: str = "summary"
    composite_score: float
    executive_summary: str
    behavioral_groups: list[dict] = []


class HealthStatus(BaseModel):
    ollama: dict
    huggingface: dict


class ModelListResponse(BaseModel):
    models: list[str]
    active: str


class ModelSwitchRequest(BaseModel):
    model: str

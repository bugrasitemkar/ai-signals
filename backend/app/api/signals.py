import asyncio
import json
from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from app.signals.engine import compute_fast_signals, compute_layer_signals, compute_behavioral_signals
from app.signals.calibration import compute_composite_score, compute_confidence_agreement
from app.summary import generate_executive_summary

router = APIRouter()

_request_store: dict[str, dict] = {}


def store_request_data(request_id: str, data: dict):
    _request_store[request_id] = data
    if len(_request_store) > 20:
        oldest = list(_request_store.keys())[0]
        del _request_store[oldest]


@router.get("/api/signals/stream")
async def signal_stream(request_id: str = Query(...)):
    if request_id not in _request_store:
        async def error_stream():
            yield {"event": "error", "data": json.dumps({"error": "Request not found"})}
        return EventSourceResponse(error_stream())

    data = _request_store[request_id]

    async def event_generator():
        all_signals = {}

        # Phase 1: Fast signals (instant)
        fast = compute_fast_signals(
            data.get("logprobs", []),
            data.get("top_logprobs"),
            data.get("response", ""),
        )
        for sig_id, sig_data in fast.items():
            all_signals[sig_id] = sig_data
            yield {"event": "signal", "data": json.dumps(sig_data)}
        await asyncio.sleep(0.1)

        # Phase 2: Layer-wise + Geometric signals (3-5s)
        try:
            layer_signals = await asyncio.get_event_loop().run_in_executor(
                None, compute_layer_signals, data.get("question", "")
            )
            for sig_id, sig_data in layer_signals.items():
                all_signals[sig_id] = sig_data
                yield {"event": "signal", "data": json.dumps(sig_data)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": f"Layer signals failed: {str(e)}"})}
        await asyncio.sleep(0.1)

        # Phase 3: Behavioral signals (5-8s)
        try:
            behavioral = await compute_behavioral_signals(data.get("question", ""))
            for sig_id, sig_data in behavioral.items():
                all_signals[sig_id] = sig_data
                yield {"event": "signal", "data": json.dumps(sig_data)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": f"Behavioral signals failed: {str(e)}"})}
        await asyncio.sleep(0.1)

        # Phase 4: Composite score
        composite = compute_composite_score(all_signals)
        all_signals[composite["signal_id"]] = composite
        yield {"event": "signal", "data": json.dumps(composite)}

        # Phase 5: Executive summary
        try:
            summary_text = await generate_executive_summary(
                data.get("question", ""),
                data.get("response", ""),
                all_signals,
            )
            summary = {
                "type": "summary",
                "composite_score": composite["value"],
                "executive_summary": summary_text,
            }
            yield {"event": "summary", "data": json.dumps(summary)}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": f"Summary failed: {str(e)}"})}

        yield {"event": "done", "data": json.dumps({"status": "complete"})}

    return EventSourceResponse(event_generator())

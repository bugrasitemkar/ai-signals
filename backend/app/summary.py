from app.models.ollama_client import generate_summary


async def generate_executive_summary(
    question: str,
    response: str,
    signals: dict,
) -> str:
    """Generate an executive summary interpreting all computed signals."""
    signal_lines = []
    for sig_id, sig_data in signals.items():
        if isinstance(sig_data.get("value"), (int, float)):
            signal_lines.append(f"- {sig_id}: {sig_data['value']} ({sig_data.get('interpretation', '')})")

    signal_text = "\n".join(signal_lines)

    prompt = f"""You are analyzing an LLM's response signals. Write a brief executive summary (2-3 sentences) interpreting what the signals collectively mean about the response's reliability.

Question asked: "{question}"
Response given: "{response[:500]}"

Computed signals:
{signal_text}

Write a concise paragraph referencing specific signal names and values. Focus on what a learner should pay attention to. Do not use bullet points."""

    return await generate_summary(prompt)

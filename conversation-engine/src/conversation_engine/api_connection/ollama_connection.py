from datetime import UTC, datetime

from langfuse import get_client
from ollama import AsyncClient

client = AsyncClient(host="http://localhost:11434")

# Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL from the environment
langfuse = get_client()

async def generate_answer(
    model: str,
    messages: str,
):
    # Not a "current" observation: the OTel context would not survive across the
    # yields of a generator consumed by StreamingResponse, so the span is ended manually
    generation = langfuse.start_observation(
        name="ollama-generate",
        as_type="generation",
        model=model,
        model_parameters={"think": False},
        input=messages,
    )
    answer = []

    try:
        response = await client.generate(
            model=model,
            prompt=messages,
            stream=True,
            think=False
        )

        async for chunk in response:
            content = chunk["response"]

            if content:
                if not answer:
                    generation.update(completion_start_time=datetime.now(UTC))
                answer.append(content)
                yield content

            if chunk.get("done"):
                generation.update(
                    usage_details={
                        "input": chunk.get("prompt_eval_count") or 0,
                        "output": chunk.get("eval_count") or 0,
                    },
                    metadata={"done_reason": chunk.get("done_reason")},
                )
    except Exception as error:
        generation.update(level="ERROR", status_message=str(error))
        raise
    finally:
        # Also reached when the client disconnects mid-stream (GeneratorExit)
        generation.update(output="".join(answer))
        generation.end()

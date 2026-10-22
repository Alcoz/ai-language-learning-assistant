from datetime import UTC, datetime

from langfuse import LangfuseGeneration, get_client, propagate_attributes
from ollama import AsyncClient

client = AsyncClient(host="http://localhost:11434")
langfuse = get_client()

def start_generation_trace(
    model: str,
    prompt: str,
    session_id: str | None,
) -> LangfuseGeneration:
    with propagate_attributes(session_id=session_id):
        return langfuse.start_observation(
            name="ollama-generate",
            as_type="generation",
            model=model,
            model_parameters={"think": False},
            input=prompt,
        )

async def generate_answer(
    model: str,
    messages: str,
    session_id: str | None = None,
):
    generation = start_generation_trace(model, messages, session_id)
    answer_chunks = []

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
                if not answer_chunks:
                    generation.update(completion_start_time=datetime.now(UTC))
                answer_chunks.append(content)
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
        generation.update(output="".join(answer_chunks))
        generation.end()

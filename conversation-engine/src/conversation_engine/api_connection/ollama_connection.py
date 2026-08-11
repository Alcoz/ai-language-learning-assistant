from ollama import generate
from ollama import AsyncClient

client = AsyncClient(host="http://localhost:11434")

async def generate_answer(
    model: str,
    messages: str,
):
    response = await client.generate(
        model=model,
        prompt=messages,
        stream=True,
        think=False
        
    )

    async for chunk in response:
        content = chunk["response"]

        if content:
            yield content
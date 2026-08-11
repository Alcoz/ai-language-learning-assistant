from ollama import generate
from ollama import GenerateResponse

def run_ollama_query(
    model: str,
    messages: str,

):
    response: GenerateResponse = generate(
        model=model,
        prompt=messages
    ).response

    return response
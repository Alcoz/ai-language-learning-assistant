from ollama import chat
from ollama import ChatResponse

def run_ollama_query(
    model: str,
    messages: str,

):
    response: ChatResponse = chat(
        model=model,
        messages=messages
    )

    return response
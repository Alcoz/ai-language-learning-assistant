from fastapi import FastAPI
from pydantic import BaseModel

from conversation_engine.api_connection.ollama_connection import run_ollama_query 

class Query(BaseModel):
    messages: str

def run_server():
    app = FastAPI()

    @app.post("/query")
    def query(query_content: Query):

        with open("conversation-engine/src/conversation_engine/instruction.txt") as instruction_file:
            instruction = instruction_file.read()

        prompt = instruction.format(
            mother_tongue="french",
            language="italian",
            language_level="A1",
            history=query_content.messages
        )

        llm_response = run_ollama_query(
            model="qwen3.5",
            messages=prompt
        )

        return {"response" : llm_response}

    return app
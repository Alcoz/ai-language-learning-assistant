from fastapi import FastAPI
from pydantic import BaseModel

from converstaion_engine.api_connection.ollama_connection import run_ollama_query 

class Query(BaseModel):
    messages: str


def run_server():
    app = FastAPI()


    @app.post("/query")
    def query(query_content: Query):
        print(query_content.messages)

        llm_response = run_ollama_query(
            model="qwen3.5",
            messages=[{
                'role': 'user',
                'content': query_content.messages,
            }]
        )

        return {"response" : llm_response.message.content}

    return app
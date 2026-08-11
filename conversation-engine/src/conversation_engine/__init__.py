from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from conversation_engine.api_connection.ollama_connection import generate_answer 

class Query(BaseModel):
    messages: str

def run_server():
    app = FastAPI()

    @app.post("/query")
    async def query(query_content: Query):

        with open("conversation-engine/src/conversation_engine/instruction.txt") as instruction_file:
            instruction = instruction_file.read()

        prompt = instruction.format(
            mother_tongue="french",
            language="italian",
            language_level="A1",
            history=query_content.messages
        )

        print(prompt)

        return StreamingResponse(
            generate_answer(
                model="qwen3.5:4b",
                messages=prompt
            ),
            media_type="text/plain",
        )

    return app
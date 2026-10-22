from dotenv import load_dotenv

# Must run before importing conversation_engine: the Langfuse client reads its keys at import
load_dotenv()

import uvicorn
from conversation_engine import run_server

if __name__ == "__main__":
    uvicorn.run(
        run_server(), host="0.0.0.0", port=8000
    )

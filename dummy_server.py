"""
Simple dummy FastAPI server for testing.

- Provides a `/generate` endpoint that returns a mock text response for a given prompt.
- Supports local testing and load testing without running a real model.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate(request: PromptRequest):
    return JSONResponse(content={"text": f"Generated response for: {request.prompt}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

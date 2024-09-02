from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import litellm

app = FastAPI()


# Pydantic models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 8192


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(
    request: ChatCompletionRequest,
    authorization: str = Header(..., description="Bearer token"),
):
    # Extract the API key from the Authorization header
    api_key = authorization.split("Bearer ")[-1]

    try:
        # Configure LiteLLM
        litellm.set_verbose = True

        # Map OpenAI model names to Anthropic model names
        model_mapping = {
            "gpt-3.5-turbo": "claude-3-sonnet-20240229",
            "gpt-4": "claude-3-5-sonnet-20240620",
        }

        anthropic_model = model_mapping.get(request.model, "gpt-4")

        # Convert Pydantic Message objects to dictionaries
        messages = [message.dict() for message in request.messages]

        # Call Anthropic API via LiteLLM
        response = litellm.completion(
            model=anthropic_model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            api_key=api_key,  # Use the API key passed from Promptflow
        )

        # Format response to match OpenAI structure
        return ChatCompletionResponse(
            id=response.id,
            object="chat.completion",
            created=response.created,
            model=request.model,  # Return the requested model name
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }
            ],
            usage=response.usage.dict(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6969)

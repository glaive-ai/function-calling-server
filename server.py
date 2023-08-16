import argparse
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM

from server_types import ChatCompletion, ChatInput, Choice
from inference import generate_message

app = FastAPI(title="Glaive Function API")


@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_endpoint(chat_input: ChatInput):
    
    request_id = str(uuid.uuid4())
    response_message = generate_message(
        messages=chat_input.messages,
        functions=chat_input.functions,
        temperature=chat_input.temperature,
        model=model,  # type: ignore
        tokenizer=tokenizer,
    )

    return ChatCompletion(
        id=request_id, choices=[Choice.from_message(response_message)]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glaive Function API")
    parser.add_argument(
        "--model",
        type=str,
        default="glaiveai/glaive-function-calling-v2-small",
        help="Model name",
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)

    uvicorn.run(app, host="0.0.0.0", port=8000)
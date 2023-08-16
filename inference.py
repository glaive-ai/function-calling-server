from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from server_types import FunctionCall, Function, ChatMessage

import demjson3 as json

def prepare_messages_for_inference(
    tokenizer: AutoTokenizer, messages: List[ChatMessage], functions=None
) -> torch.Tensor:
    conversation = []
    
    if functions is not None:
        function_defs = [funnction_def.model_dump_json(indent=4) for funnction_def in functions]

    system_message = next(filter(lambda message: message.role == "system", messages), None)
    if system_message is None:
        if functions is None:
            system_prompt = "SYSTEM: You are a helpful assistant, with no access to external functions."
        else:
            system_prompt = "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n" + "\n".join(function_defs)
    else:
        if functions is None:
            system_prompt = "SYSTEM: " + system_message.content
        else:
            system_prompt = "SYSTEM: " + system_message.content + "\n"+ "\n".join(function_defs)
    conversation.append(system_prompt)
    for message in messages:
        if message.role == "user":
            user_message = "USER: "+message.content
            conversation.append(user_message)
        if message.role == "assistant":
            if message.function_call:
                assistant_function_call = "ASSISTANT: <functioncall> " + message.function_call.model_dump_json()
                conversation.append(assistant_function_call)
            if message.content:
                assistant_message = "ASSISTANT: " + message.content
                conversation.append(assistant_message)
        if message.role == "function":
            function_response = "FUNCTION RESPONSE: " + message.content
            conversation.append(function_response)

    conversation = "\n".join(conversation)
    input_ids = tokenizer(conversation,return_tensors="pt")
    return input_ids,conversation


def generate_message(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[ChatMessage],
    functions: Optional[List[Function]] = None,
    temperature: float = 0.7,
    max_new_tokens=256,
) -> ChatMessage:
    
    inputs,conversation = prepare_messages_for_inference(
        tokenizer=tokenizer, messages=messages, functions=functions
    )
    inputs.to(model.device)
    generated_ids = model.generate(
        **inputs, do_sample = True,max_new_tokens=max_new_tokens, temperature=temperature
    )
    generated_content = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).replace(conversation,"").strip()

    # If it's a function call:
    if "<functioncall>" in generated_content:
        function_call_content = generated_content.split("<functioncall>")[1].strip()
        function_call_json = json.decode(function_call_content)
        function_name = function_call_json["name"]
        arguments = function_call_json["arguments"]
        return ChatMessage(
            role="assistant",
            function_call=FunctionCall(name=function_name, arguments=arguments),
        )
    return ChatMessage(
        role="assistant",
        content=generated_content,
    )
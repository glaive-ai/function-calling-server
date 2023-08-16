import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "glaive" # We just need to set this something other than None, so it works with openai package. No API key is required.

print(openai.ChatCompletion.create(
    model="glaive",
    messages=[{"role": "user", "content": "What is the weather for Istanbul?"}],
    functions=[{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
            },
            "required": ["location"],
        },
    },
    ]
))
import os
from openai import OpenAI
from dotenv import dotenv_values


config = dotenv_values(".env")

# Creating the model to be used
client = OpenAI(api_key=config["OPENAI_KEY"])

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, my name is Manilson."}
    ]
)

print(completion.choices[0].message)
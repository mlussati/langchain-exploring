from langchain_openai import OpenAI
from dotenv import dotenv_values

config = dotenv_values(".env")


# Creating the model to be used
llm = OpenAI(api_key=config["OPENAI_KEY"])

response = llm.invoke("What is the tallest building in the world?")
print(response)
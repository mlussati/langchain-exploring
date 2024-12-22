from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from dotenv import dotenv_values

config = dotenv_values("/Users/manilsonlussati/Documents/workspace/langchain-exploring/.env")

# Creating the model to be used
# llm = OpenAI(api_key=config["OPENAI_KEY"])

# response = llm.invoke("What is the tallest building in the world?")
# print(response)

#SYSTEM_MESSAGE
chat = ChatOpenAI(api_key=config["OPENAI_KEY"])

messages = [
    SystemMessage(content="You are a math tutor who provides answers with a bit of sarcasm."),
    HumanMessage(content="What is the square of 2?"),
]

response = chat.invoke(messages)
print(response.content)

#PROMPTS

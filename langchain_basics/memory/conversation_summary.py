from langchain_openai import OpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from dotenv import dotenv_values
config = dotenv_values("/Users/manilsonlussati/Documents/workspace/langchain-exploring/.env")

llm = OpenAI(temperature=0.0, openai_api_key=config["OPENAI_KEY"])

memory = ConversationSummaryMemory(llm=llm)
memory.save_context({"input": "Alex is a 9-year old boy."}, 
                    {"output": "Hello Alex! How can I assist you today?"})
memory.save_context({"input": "Alex likes to play football"}, 
                    {"output": "That's great to hear! "})

conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)

print(conversation.predict(input="How old is Alex?"))
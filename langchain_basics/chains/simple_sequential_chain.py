#importing the modules
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from dotenv import dotenv_values
config = dotenv_values("/Users/manilsonlussati/Documents/workspace/langchain-exploring/.env")

llm = OpenAI(temperature=0.0, openai_api_key=config["OPENAI_KEY"])

prompt_1 = PromptTemplate(
    input_variables=["book"],
    template="Name the author of the book {book}?",
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = PromptTemplate(
    input_variables=["author_name"],
    template="Write a 50-word biography of the following author: {author_name}",
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

simple_sequential_chain = SimpleSequentialChain(
    chains=[chain_1, chain_2],
    verbose=True
)

simple_sequential_chain.run("Alice's Adventures in Wonderland")
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import dotenv_values

config = dotenv_values("/Users/manilsonlussati/Documents/workspace/langchain-exploring/.env")

llm = OpenAI(temperature=0.0, openai_api_key=config["OPENAI_KEY"])

prompt_template = PromptTemplate(
    input_variables=["film"],
    template="Name the author of the film {film}?",
)

chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True)

print(chain.run("Interstellar"))

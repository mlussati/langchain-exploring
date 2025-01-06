# Example: Getting information from an external API

from langchain_openai import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from dotenv import dotenv_values

config = dotenv_values("/Users/manilsonlussati/Documents/workspace/langchain-exploring/.env")

llm = OpenAI(temperature=0.0, openai_api_key=config["OPENAI_KEY"])
serpapi_api_key = config.get("SERPAPI_API_KEY")

# Loading Tools
tools = load_tools(["serpapi",
                    "llm-math"],
                    llm=llm,
                    serpapi_api_key=serpapi_api_key)

agent = initialize_agent(
                        tools,
                        llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True)

print(agent.run("What is the current population of the world, and calculate the percentage change compared to the population five years ago"))
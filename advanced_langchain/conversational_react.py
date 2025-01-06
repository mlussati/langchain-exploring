# Example: Empowering ChatGPT with current knowledge
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, load_tools
from dotenv import dotenv_values

config = dotenv_values("/Users/manilsonlussati/Documents/workspace/langchain-exploring/.env")

llm = OpenAI(temperature=0.0, openai_api_key=config["OPENAI_KEY"])
serpapi_api_key = config.get("SERPAPI_API_KEY")

memory = ConversationBufferMemory(memory_key="chat_history")

# Loading Tools
tools = load_tools(["serpapi"],
                    llm=llm,
                    serpapi_api_key=serpapi_api_key)

agent = initialize_agent(tools,
                        llm,
                        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                        verbose=True,
                        memory=memory)

agent.run("Hi, my name is Manilson, and I live in the São Paulo City.")
agent.run("My favorite game is basketball.")
print(agent.run("Give me the list of statiums to watch a basketabll game in my city today. \
                Also give the teams that are playing"))
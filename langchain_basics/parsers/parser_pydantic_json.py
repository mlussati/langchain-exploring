from typing import List
#from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import dotenv_values

config = dotenv_values("..../langchain-exploring/.env")

model = OpenAI(openai_api_key=config["OPENAI_KEY"])

class Author(BaseModel):
    number: int = Field(description="Number of books written by the author")
    books: List[str] = Field(description="List of books they wrote")

output_parser = PydanticOutputParser(pydantic_object=Author)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

user_query = "Generate the books written by Dan Brown."

formatted_prompt = prompt.format(query=user_query)

response = model.invoke(formatted_prompt)

parsed_response = output_parser.parse(response)

print(parsed_response)

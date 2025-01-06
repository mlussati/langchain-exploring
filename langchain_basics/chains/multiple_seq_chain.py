#importing the modules
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from dotenv import dotenv_values

from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain

config = dotenv_values("/Users/manilsonlussati/Documents/workspace/langchain-exploring/.env")

llm = OpenAI(temperature=0.0, openai_api_key=config["OPENAI_KEY"])

biography = "He is an American author of thriller fiction, best known for his Robert Langdon series. \
          He has sold over 200 million copies of his books, which have been translated into 56 \
          languages. His other works include Angels & Demons, The Lost Symbol, Inferno, and Origin. \
          He is a New York Times best-selling author and has been awarded numerous awards for his \
          writing."

prompt_1 = ChatPromptTemplate.from_template(
    "Summarize this biography in one sentence:"
    "\n\n{biography}"
)

chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="one_line_biography")

prompt_2 = ChatPromptTemplate.from_template(
    "Can you tell the author's name in this biography:"
    "\n\n{one_line_biography}"
)

chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="author_name")

prompt_3 = ChatPromptTemplate.from_template(
    "Tell the name of the highest selling book of this author:"
    "\n\n{author_name}"
)

chain_3 = LLMChain(llm=llm, prompt=prompt_3, output_key="book")

prompt_4 = ChatPromptTemplate.from_template(
    "Write a follow-up response to the following "
    "summary of the highest-selling book of the author:"
    "\n\nAuthor: {author_name}\n\nBook: {book}"
)

# we input the author's name and the highest-selling book, and the output is the book's summary
chain_4 = LLMChain(llm=llm, prompt=prompt_4, output_key="summary")

final_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3, chain_4],
    input_variables=["biography"],
    output_variables=["one_line_biography", "author_name", "summary"],
    verbose=True
)

outputs = final_chain.invoke({"biography": biography})
print(outputs)
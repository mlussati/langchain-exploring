# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate, FewShotPromptTemplate
# from langchain.schema.messages import HumanMessage, SystemMessage
# from dotenv import dotenv_values

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import dotenv_values

config = dotenv_values(".../langchain-exploring/.env")

# Creating the model to be used
# llm = OpenAI(api_key=config["OPENAI_KEY"])

# response = llm.invoke("What is the tallest building in the world?")
# print(response)

#SYSTEM_MESSAGE
chat = ChatOpenAI(temperature=0.0, api_key=config["OPENAI_KEY"])

# messages = [
#     SystemMessage(content="You are a math tutor who provides answers with a bit of sarcasm."),
#     HumanMessage(content="What is the square of 2?"),
# ]

# response = chat.invoke(messages)
# print(response.content)

#PROMPTS
# email_template = PromptTemplate.from_template(
#     "Create an invitation email to the recipinet that is {recipient_name} \
#  for an event that is {event_type} in a language that is {language} \
#  Mention the event location that is {event_location} \
#  and event date that is {event_date}. \
#  Also write few sentences about the event description that is {event_description} \
#  in style that is {style} "
# )

# message = email_template.format(
#     style="enthusiastic",
#     language="American English",
#     recipient_name="John",
#     event_type="product launch",
#     event_date="January 15, 2024",
#     event_location="Grand Ballroom, City Center Hotel",
#     event_description="an exciting unveiling of our latest innovations",
# )

# response = llm.invoke(message)
# print(response)

#FEW-SHOT TEMPLATE
examples = [
    {
        "review": "I absolutely love this product! It exceeded my expectations.",
        "sentiment": "Positive"
    },
    {
        "review": "I'm really disappointed with the quality of this item. It didn't meet my needs.",
        "sentiment": "Negative"
    },
    {
        "review": "The product is okay, but there's room for improvement.",
        "sentiment": "Neutral"
    }
]

example_prompt = PromptTemplate(
    input_variables=["review", "sentiment"],
    template="Review: {review}\n{sentiment}"
)
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Review: {input}",
    input_variables=["input"]
)

message = prompt.format(input="The machine worked okay without much trouble.")
response = chat.invoke(message)
print(response.content)

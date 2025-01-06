import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

#Method HW1 method3
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

#Method HW1 method 1 & 2
#from pydantic import BaseModel, Field
from typing import Optional, Union
from typing_extensions import Annotated, TypedDict

print_with_method=1

class Holiday(TypedDict):
    """Answer the date and name of holiday."""

    date: str = Annotated[str, ..., "The date of the holiday, must be YYYY-mm-DD format"]
    name: str = Annotated[str, ..., "The short name of the holiday"]

class FinalResponse(TypedDict):
    Result: Union[Holiday]

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    responseStr=""
    if print_with_method == 1:
        print("Output with method 1")
        llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
        )
        structured_llm = llm.with_structured_output(FinalResponse)
        response = structured_llm.invoke(question)
        responseStr = json.dumps(response, ensure_ascii=False)
    elif print_with_method == 2:
        print("Output with method 2")
        llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
        )
        model_with_tools = llm.bind_tools([FinalResponse])
        response = model_with_tools.invoke(question)
        responseStr = json.dumps(response.tool_calls[0]["args"], ensure_ascii=False)
    else:
        print("Output with method 3")
        llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature'],
            model_kwargs={ "response_format": { "type": "json_object" } }
        )
        prompt = ChatPromptTemplate.from_template(
            "Answer the user's question to the best of your ability."
            'You must always output a JSON object with an "date" key which format is YYYY-mm-DD and a "name" key which is the name of holiday.'
            "{question}"
        )
        chain = prompt | llm | SimpleJsonOutputParser()
        response = chain.invoke({ "question": question })
        responseStr = json.dumps(response, ensure_ascii=False)
    return responseStr
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    return response


print(generate_hw01("2024年台灣10月紀念日有哪些?"))

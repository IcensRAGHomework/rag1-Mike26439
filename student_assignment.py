import json
import re
import traceback

from model_configurations import get_model_configuration

from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from typing import List
from pydantic import BaseModel, Field

class Holiday(BaseModel):
    """The date and name of holiday."""

    date: str = Field(..., description="The date of the holiday, must be YYYY-mm-DD format")
    name: str = Field(..., description="The short name of the holiday")

class Result(BaseModel):
    Result: List[Holiday]

# Custom parser
def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between \`\`\`json and \`\`\` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"\`\`\`json(.*?)\`\`\`"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")

# Set up a parser
parser = PydanticOutputParser(pydantic_object=Result)

####################################################
gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    responseStr=""
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user question."
            "Output as JSON without description."
            "With one Result object which conatins multiple object with date key and event key."
        ),
        ("human", "{question}"),
    ]
    )
    #print(prompt.format_prompt(question=question).to_string())

    chain = prompt | llm
    response = chain.invoke({ "question": question })
    responseStr = json.dumps(extract_json(response)[0], ensure_ascii=False)
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


#print(demo("2024年台灣10月紀念日有哪些?"))
#generate_hw01("2024年台灣10月紀念日有哪些?")
#print(generate_hw01("2024年台灣10月紀念日有哪些?"))

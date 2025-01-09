import json
import re
import traceback
import sys

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
            "Please provide the only one anniversaries in Taiwan for the specified month and year."
            "Output as JSON without description."
            "Only return the JSON content with date key and name key."
            #"Return object "
        ),
        ("human", "{question}"),
    ]
    )
    print(prompt.format_prompt(question=question).to_string())

    chain = prompt | llm
    response = chain.invoke({ "question": question })
    content = extract_json(response)[0]

    if len(sys.argv) == 2 and sys.argv[1] == "1":
        print(response)
        print("\n")
        print(content)
        responseStr = json.dumps({"Result": content}, ensure_ascii=False)
        #responseStr = json.dumps(extract_json(response)[0], ensure_ascii=False)
    else:
        print(response)
        print("\n")
        print(content)
        responseStr = json.dumps({"Result": content})
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

if len(sys.argv) == 2:
    if sys.argv[1] == "1":
        print(generate_hw01("2024年台灣10月紀念日有哪些?"))
    elif  sys.argv[1] == "2":
        pass
    elif  sys.argv[1] == "3":
        pass
    elif  sys.argv[1] == "4":
        pass
#else:
    #print(demo("2024年台灣10月紀念日有哪些?"))
    #print(generate_hw01("2024年台灣10月紀念日有哪些?"))

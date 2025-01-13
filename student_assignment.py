import json
import re
import traceback
import sys

from model_configurations import get_model_configuration
from model_configurations import get_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

#import for json parser
from langchain_core.output_parsers import JsonOutputParser

#import for class
from typing import List
from pydantic import BaseModel, Field

class Holiday(BaseModel):
    """Information about a holiday."""

    date: str = Field(..., description="The date of the holiday, must be YYYY-mm-DD format")
    name: str = Field(..., description="The short name of the holiday")

class Result(BaseModel):
    """Identifying information about all holidays in specific month."""
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
parser = JsonOutputParser(pydantic_object=Result)

####################################################
gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
calendarific_api_key = (
    get_configuration("CALENDARIFIC_API_KEY") or "JnNVNp1K8XdG11hYyGnJUfM3edOwilMd"
)

llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

def generate_hw01(question):
    responseStr=""

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user question."
            "Output as JSON without description."
            "Please provide the anniversaries in Taiwan for the specified month and year in JSON format"
            "matches the given schema: \`\`\`json\n{schema}\n\`\`\`. "
            "Make sure to wrap the answer in \`\`\`json and \`\`\` tags",
        ),
        ("human", "{question}"),
    ]
    ).partial(schema=Result.model_json_schema())
    #print(prompt.format_prompt(question=question).to_string())

    chain = prompt | llm | parser
    response = chain.invoke({ "question": question })
    responseStr = json.dumps(response, ensure_ascii=False)
    if sys.argv[1] == "1":
        print(response)
        print(responseStr)

    return responseStr

def generate_hw02(question):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
        ]
    )

    functions = [
        {
            "name": "get_anniversaries_from_api",
            "description": "Fetches the anniversaries for a specified country in ISO-3166 format, month, and year using the Calendarific API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "month": {"type": "integer"},
                    "year": {"type": "integer"},
                },
                "required": ["country", "month", "year"],
            }
        }
    ]
    model_with_function = llm.bind(functions=functions)
    chain = prompt | model_with_function
    response = chain.invoke({ "question": question })
    additional_kwargs = json.loads(json.dumps(response.additional_kwargs))
    args = json.loads(additional_kwargs['function_call']['arguments'])
    year = args['year']
    month = args['month']
    country = args['country']
    data = get_anniversaries_from_api(country, month, year)
    print(data["response"]["holidays"])

    if True:
        # Convert and filter fetch data to json format in homework style
        response_array = []
        for item in data["response"]["holidays"]:
            item_object = {
                    'date': item["date"]["iso"],
                    'name': item["name"],
            }
            response_array.append(item_object)

        #final_response_json = json.dumps(response_array)
        #final_response_json = "{ \"Result\": " + final_response_json + " }"
        return json.dumps({"Result": response_array})
    else:
        anniversaries = data.get("response").get("holidays")
        holidays = [
            {"date": holiday["date"]["iso"], "name": holiday["name"]}
            for holiday in anniversaries
        ]
        return json.dumps({"Result": holidays})

def get_anniversaries_from_api(country, month, year):
    import requests

    url = f"https://calendarific.com/api/v2/holidays?api_key={calendarific_api_key}&country={country}&language=zh&year={year}&month={month}"
    response = requests.get(url)
    return response.json()

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

#print(sys.argv)
#print(demo("2024年台灣10月紀念日有哪些?"))
#generate_hw01("2024年台灣10月紀念日有哪些?")
#print(generate_hw01("2024年台灣10月紀念日有哪些?"))


if len(sys.argv) == 2:
    if sys.argv[1] == "1":
        print(generate_hw01("2024年台灣10月紀念日有哪些?"))
    elif  sys.argv[1] == "2":
        print(generate_hw02("2024年台灣10月紀念日有哪些?"))
    elif  sys.argv[1] == "3":
        generate_hw03("2024年台灣10月紀念日有哪些?", "根據先前的節日清單，這個節日{\"date\": \"10-31\", \"name\": \"蔣公誕辰紀念日\"}是否有在該月份清單？")
    elif  sys.argv[1] == "4":
        generate_hw01("請問中華台北的積分是多少")
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"It is sunny in {city}"

llm = ChatAnthropic(model="claude-sonnet-4-6")
llm_with_tools = llm.bind_tools([get_weather])

messages = [HumanMessage("What is the weather in New York?")]
response = llm_with_tools.invoke(messages)

# Execute any tool calls and send results back to the model
if response.tool_calls:
    messages.append(response)
    for tool_call in response.tool_calls:
        result = get_weather.invoke(tool_call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    final = llm_with_tools.invoke(messages)
    print(final.content)
else:
    print(response.content)

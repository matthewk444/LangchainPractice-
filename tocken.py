from langgraph.graph import Stateraph, End 
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage 
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
import os
load_dotenv()

# 1 State graph definition
class AgentState(TypedDict): 
    messages: Annotated[list, operator.add]

#2 Define tools 
@tool
#create a tool to count tokens or something 

#3 LLM 
llm = ChatAnthropic(
    api_key = os.getenv("ANTHROPIC_API_KEY"),
    model = "claude-sonnet-4-6"
)




    


from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate 
from dotenv import load_dotenv
import os
load_dotenv

# Task is to create an agent to answer questions about mlb player stats. 

# ! Define tools 
@tool 
def get_player_stats(player_name: str) -> str:
    """Get current stats for given MLB player."""
    return f"Current stats for {player_name}: 30 Home Runs, 100 RBIs, .270 BA, .456 OBP, .665 SLG" 

@tool 
def compare_players(player1: str, player2: str) -> str:
    """Compare stats of two MLB players."""
    return f"Comparing {player1} and {player2}: {player1} has better power, {player2} has better contact."

# 2 set up LLM and agent 
llm  = ChatAnthropic(
    api_key = os.getenv("ANTHROPIC_API_KEY"),
    model = "claude-sonnet-4-6"
)

# 3 Prompt 

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides MLB player stats."),
    ("human", "{input}"), 
    ("placeholder", "{agent_scratchpad}")


])

#4 Create the agent 

tools = [get_player_stats, compare_players]
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools) # Agent executor invisibly sends results back to the model then our LLM creates its final answer. 

#5 run it 

result = executor.invoke({"input": "Compare Aaron Judge and Shohei Ohtani's stats with the data to prove it. "})
print(result["output"])
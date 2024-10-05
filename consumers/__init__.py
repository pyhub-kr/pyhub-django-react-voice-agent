from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults

from .base import BaseOpenAIRealtimeConsumer


@tool
def add(a: int, b: int):
    """Add two numbers. Please let the user know that you're adding the numbers BEFORE you call the tool"""
    return a + b


tavily_tool = TavilySearchResults(
    max_results=5,
    include_answer=True,
    description=(
        "This is a search tool for accessing the internet.\n\n"
        "Let the user know you're asking your friend Tavily for help before you call the tool."
    ),
)


class OpenAIRealtimeConsumer(BaseOpenAIRealtimeConsumer):
    instructions = "You are a helpful assistant. Speak Korean."
    tools = [add, tavily_tool]

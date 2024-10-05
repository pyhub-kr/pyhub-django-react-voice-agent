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

    async def check_permission(self, user) -> bool:
        """웹소켓 연결을 요청받았을 때 호출됩니다. 사용자의 권한을 확인하며 거짓을 반환하면 웹소켓 연결 요청을 거부합니다."""
        # 로그인 여부로 웹소켓 접속 권한을 체크할 경우
        # return await sync_to_async(lambda: user.is_authenticated)()
        return True

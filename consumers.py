import asyncio

from channels.generic.websocket import AsyncWebsocketConsumer

from langchain_openai_voice import OpenAIVoiceReactAgent
from tools import TOOLS


class RealtimeConsumer(AsyncWebsocketConsumer):
    agent: OpenAIVoiceReactAgent
    input_queue: asyncio.Queue
    agent_task: asyncio.Task

    async def connect(self):
        await self.accept()
        self.agent = OpenAIVoiceReactAgent(
            model="gpt-4o-realtime-preview",
            tools=TOOLS,
            instructions="You are a helpful assistant. Speak Korean.",
        )
        self.input_queue = asyncio.Queue()
        self.agent_task = asyncio.create_task(self.run_agent())
        self.agent_task.add_done_callback(self.handle_agent_task_result)

    @staticmethod
    def handle_agent_task_result(task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Agent task failed with exception: {e}")

    async def disconnect(self, code):
        # Cancel the agent task if it's still running
        if hasattr(self, "agent_task") and not self.agent_task.done():
            self.agent_task.cancel()
            try:
                await self.agent_task
            except asyncio.CancelledError:
                pass  # This is expected

    async def receive(self, text_data=None, bytes_data=None):
        await self.input_queue.put(text_data)

    async def run_agent(self):
        async def input_stream():
            while True:
                yield await self.input_queue.get()

        async def send_output_chunk(chunk):
            await self.send(text_data=chunk)

        await self.agent.aconnect(
            input_stream=input_stream(),
            send_output_chunk=send_output_chunk,
        )

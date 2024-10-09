import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Union, Literal

import websockets
from langchain_core.tools import BaseTool

from django.contrib.auth import get_user_model

from langchain_openai_voice import (
    amerge,
    VoiceToolExecutor,
)

from channels.generic.websocket import AsyncWebsocketConsumer


logger = logging.getLogger(__name__)

User = get_user_model()


class BaseOpenAIRealtimeConsumer(AsyncWebsocketConsumer):

    url: str = "wss://api.openai.com/v1/realtime"
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    # https://platform.openai.com/docs/models/gpt-4o-realtime
    model: str = "gpt-4o-realtime-preview"
    input_audio_transcription_model: Literal["whisper-1"] = "whisper-1"

    voice: Literal["alloy"] = "alloy"
    instructions: str = "You are a helpful assistant."
    temperature: float = 0.8
    tools: list[BaseTool] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_queue = asyncio.Queue()

        self.agent_task: asyncio.Task | None = None
        self.openai_realtime_websocket: websockets.WebSocketClientProtocol | None = None

        tools_by_name = {tool.name: tool for tool in self.tools or []}
        self.tool_executor = VoiceToolExecutor(tools_by_name=tools_by_name)

    async def check_permission(self, user) -> bool:
        """
        웹소켓 연결을 요청받았을 때 호출됩니다. 사용자의 권한을 확인하며 거짓을 반환하면 웹소켓 연결 요청을 거부합니다.

        Args:
            user: 권한을 확인할 사용자 객체

        Returns:
            bool: 사용자가 권한이 있으면 True, 없으면 False

        Note:
            현재는 모든 사용자에게 권한을 부여합니다.
            필요에 따라 이 메서드를 수정하여 실제 권한 검사를 구현할 수 있습니다.
        """
        return True

    async def connect(self):
        """
        WebSocket 연결을 처리하는 메서드입니다.

        이 메서드는 다음과 같은 기능을 수행합니다:
        1. 사용자 인증 정보 확인
        2. 사용자 권한 검사
        3. WebSocket 연결 수락 또는 거부
        4. 에이전트 태스크 생성 및 실행

        연결 과정에서 발생할 수 있는 오류:
        - 4500: AuthMiddlewareStack이 적용되지 않은 경우
        - 4403: 사용자 권한이 없는 경우

        Returns:
            None
        """

        if "user" not in self.scope:
            logger.error(
                "사용자 정보가 scope에 없습니다. asgi.py에서 AuthMiddlewareStack을 적용해주세요."
            )
            await self.close(
                code=4500
            )  # 4500: 커스텀 에러 코드 (AuthMiddlewareStack 미적용)
            return

        if not await self.check_permission(self.scope["user"]):
            # 수락하고 종료를 해야, 지정한 종료코드가 전달됩니다.
            await self.accept()
            await self.close(
                code=4403
            )  # 4403: Forbidden (custom code for unauthorized access)
        else:
            await self.accept()
            self.agent_task = asyncio.create_task(self.run_agent())
            self.agent_task.add_done_callback(self.handle_agent_task_result)

    @staticmethod
    def handle_agent_task_result(task: asyncio.Task) -> None:
        """
        에이전트 태스크의 결과를 처리합니다.

        이 메서드는 다음과 같은 기능을 수행합니다:
        1. 태스크의 정상 완료 여부를 확인합니다.
        2. 태스크 취소 시 조용히 처리합니다.
        3. 예외 발생 시 로깅합니다.

        비동기 태스크의 생명주기를 관리하고 예기치 않은 오류를 감지하여 디버깅을 지원합니다.

        Args:
            task (asyncio.Task): 처리할 asyncio.Task 객체

        Returns:
            None
        """
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Agent task failed with exception: %s", e)

    async def disconnect(self, code: int) -> None:
        """
        현재 연결된 웹소켓 클라이언트와 WebSocket 연결이 종료될 때 호출됩니다.

        1. 실행 중인 에이전트 태스크가 있다면 취소합니다.
        2. OpenAI 실시간 WebSocket 연결이 열려있다면 닫습니다.

        Args:
            code (int): WebSocket 종료 코드

        Returns:
            None
        """
        if self.agent_task and not self.agent_task.done():
            self.agent_task.cancel()
            try:
                await self.agent_task
            except asyncio.CancelledError:
                pass
        if self.openai_realtime_websocket:
            await self.openai_realtime_websocket.close()

    async def receive(self, text_data: str = None, bytes_data: bytes = None) -> None:
        """
        클라이언트로부터 데이터를 수신하는 메서드입니다.

        이 메서드는 다음과 같은 기능을 수행합니다:
        1. 유저 음성 데이터를 텍스트 데이터로 전달받고, 입력 큐를 통해 에이전트에게 전달합니다.
        2. bytes_data는 무시합니다.

        Args:
            text_data (str, optional): 클라이언트로부터 받은 텍스트 데이터
            bytes_data (bytes, optional): 클라이언트로부터 받은 바이트 데이터 (무시됨)

        Returns:
            None
        """
        await self.input_queue.put(text_data)

    async def send_to_openai(self, event: Union[dict[str, Any], str]) -> None:
        """
        OpenAI 실시간 WebSocket으로 이벤트를 전송합니다.

        Args:
            event (Union[dict[str, Any], str]): 전송할 이벤트. 딕셔너리 또는 문자열 형태일 수 있습니다.

        Returns:
            None
        """
        formatted_event = json.dumps(event) if isinstance(event, dict) else event
        await self.openai_realtime_websocket.send(formatted_event)

    async def openai_output_stream(self) -> AsyncIterator[Union[str, dict[str, Any]]]:
        """
        OpenAI 실시간 WebSocket으로부터 이벤트를 수신합니다.

        Returns:
            AsyncIterator[Union[str, dict[str, Any]]]: 이벤트 스트림
        """
        async for raw_event in self.openai_realtime_websocket:
            yield json.loads(raw_event)

    async def user_input_stream(self) -> AsyncIterator[str]:
        """
        유저 음성 데이터를 큐에서 꺼내 반환합니다.

        Returns:
            AsyncIterator[str]: 유저 음성 데이터 스트림
        """
        while True:
            yield await self.input_queue.get()

    async def on_user_mic_input(self, event) -> None:
        """사용자의 마이크 입력이 들어왔을 때, OpenAI로 사용자 입력 데이터를 전송"""
        await self.send_to_openai(event)

    async def on_tools_outputs(self, event):
        """도구 실행 결과가 나왔을 때, OpenAI로 도구 실행 결과를 전송하고, OpenAI에 새로운 응답 생성 요청"""
        logger.debug("tool output (send to openai): %s", event)
        await self.send_to_openai(event)
        # https://platform.openai.com/docs/api-reference/realtime-client-events/response-create
        await self.send_to_openai({"type": "response.create", "response": {}})

    async def run_agent(self) -> None:
        """
        OpenAI 실시간 WebSocket 연결을 설정하고, 에이전트를 실행합니다.

        Returns:
            None
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        url = f"{self.url}?model={self.model}"

        async with websockets.connect(
            url, extra_headers=headers
        ) as self.openai_realtime_websocket:
            tool_defs = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": tool.args},
                }
                for tool in self.tool_executor.tools_by_name.values()
            ]
            # https://platform.openai.com/docs/api-reference/realtime-client-events/session-update
            # 세션 디폴트 설정을 업데이트할 목적으로 이벤트 전송
            await self.send_to_openai(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": self.instructions,
                        "input_audio_transcription": {
                            "model": self.input_audio_transcription_model,
                        },
                        "voice": self.voice,
                        "turn_detection": {
                            "type": "server_vad",  # "server_vad" 만 지원
                            "threshold": 0.5,  # Activation threshold for VAD (0.0 to 1.0)
                            "prefix_padding_ms": 300,  # speech 시작 전에 적용할 audio padding
                            "silence_duration_ms": 200,  # speech stop 탐지 기준이 되는 침묵 시간 (default: 200)
                        },
                        "temperature": self.temperature,
                        "tools": tool_defs,
                    },
                }
            )

            async for stream_key, data_raw in amerge(
                user_input_stream=self.user_input_stream(),  # 마이크 입력 스트림
                tools_output_stream=self.tool_executor.output_iterator(),  # 툴 출력 스트림
                openai_output_stream=self.openai_output_stream(),  # 모델 출력 스트림
            ):
                try:
                    data: Union[dict, str] = (
                        json.loads(data_raw) if isinstance(data_raw, str) else data_raw
                    )
                except json.JSONDecodeError:
                    logger.error("Error decoding data: %s", data_raw)
                    continue

                if stream_key == "user_input_stream":
                    await self.on_user_mic_input(data)

                elif stream_key == "tools_output_stream":
                    await self.on_tools_outputs(data)

                elif stream_key == "openai_output_stream":
                    event_type = data["type"]

                    event_type_method_name = f"on_openai_{event_type.replace('.', '_')}"
                    if hasattr(self, event_type_method_name):
                        event_handler = getattr(self, event_type_method_name)
                        if asyncio.iscoroutinefunction(event_handler):
                            await event_handler(data)
                        else:
                            event_handler(data)
                    else:
                        logger.error("UNKNOWN EVENT: %s", event_type)
                else:
                    logger.error("Unknown stream key: %s", stream_key)

    #
    # OpenAI Server Event Handlers
    #  - https://platform.openai.com/docs/guides/realtime/server-events
    #

    # https://www.jetbrains.com/help/pycharm/disabling-and-enabling-inspections.html#change-highlighting-level-for-file
    # noinspection PyMethodMayBeStatic
    async def on_openai_error(self, event: dict) -> None:
        """OpenAI API 호출에서 오류가 발생했을 때 호출됩니다."""
        logger.error("[error] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_session_created(self, event: dict) -> None:
        """OpenAI 세션이 생성되었을 때 호출됩니다. 새로운 연결이 설정될 때 자동으로 발생합니다."""
        logger.debug("[session.created] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_session_updated(self, event: dict) -> None:
        """OpenAI 세션이 업데이트되었을 때 호출됩니다."""
        logger.debug("[session.updated] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_conversation_created(self, event: dict) -> None:
        """OpenAI 대화가 생성되었을 때 호출됩니다. 세션 생성 직후에 발생합니다."""
        logger.debug("[conversation.created] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_input_audio_buffer_committed(self, event: dict) -> None:
        """OpenAI 입력 오디오 버퍼가 커밋되었을 때 호출됩니다. 클라이언트에 의해 또는 서버 VAD 모드에서 자동으로 발생할 수 있습니다."""
        logger.debug("[input_audio_buffer.committed] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_input_audio_buffer_cleared(self, event: dict) -> None:
        """OpenAI 입력 오디오 버퍼가 클라이언트에 의해 지워졌을 때 호출됩니다."""
        logger.debug("[input_audio_buffer.cleared] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_input_audio_buffer_speech_started(self, event: dict) -> None:
        """Returned in server turn detection mode when speech is detected."""
        logger.debug("[input_audio_buffer.speech_started] %s", event)
        # WebSocket 클라이언트로 음성 시작 신호 전송
        await self.send(text_data=json.dumps(event))

    # noinspection PyMethodMayBeStatic
    async def on_openai_input_audio_buffer_speech_stopped(self, event: dict) -> None:
        """서버 턴 감지 모드에서 음성이 멈추었을 때 호출됩니다."""
        logger.debug("[input_audio_buffer.speech_stopped] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_conversation_item_created(self, event: dict) -> None:
        """대화 항목이 생성되었을 때 호출됩니다."""
        logger.debug("[conversation.item.created] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_conversation_item_input_audio_transcription_completed(
        self, event: dict
    ) -> None:
        """입력 오디오 음성-텍스트 변환이 활성화되어 있고 변환이 성공했을 때 실행됩니다"""
        logger.debug(
            "[conversation.item.input_audio_transcription.completed] %s", event
        )

    # noinspection PyMethodMayBeStatic
    async def on_openai_conversation_item_input_audio_transcription_failed(
        self, event: dict
    ) -> None:
        """입력 오디오 음성-텍스트 변환이 설정되어 있고, 사용자 메시지에 대한 음성-텍스트 변환 요청이 실패했을 때 호출됩니다."""
        logger.debug("[conversation.item.input_audio_transcription.failed] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_conversation_item_truncated(self, event: dict) -> None:
        """이전 어시스턴트 오디오 메시지 항목이 클라이언트에 의해 잘렸을 때 호출됩니다."""
        logger.debug("[conversation.item.truncated] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_conversation_item_deleted(self, event: dict) -> None:
        """대화에서 항목이 삭제되었을 때 호출됩니다."""
        logger.debug("[conversation.item.deleted] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_created(self, event: dict) -> None:
        """새로운 응답이 생성되었을 때 호출됩니다. 응답 생성의 첫 번째 이벤트로, 응답이 '진행 중' 초기 상태에 있습니다."""
        logger.debug("[response.created] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_done(self, event: dict) -> None:
        """응답 스트리밍이 완료되었을 때 호출됩니다. 최종 상태와 관계없이 항상 발생합니다."""
        logger.debug("[response.done] %s", event)

        usage = event["response"]["usage"]
        # https://openai.com/api/pricing/ (gpt-4o-realtime-preview 2024-10-06 기준)
        #  - Text : $5.00 / 1M input tokens, $20.00 / 1M output tokens
        #  - Audio : $100.00 / 1M input tokens, $200.00 / 1M output tokens

        # Token usage calculation
        if "input_token_details" in usage:
            text_input_tokens = usage["input_token_details"].get("text_tokens", 0)
            audio_input_tokens = usage["input_token_details"].get("audio_tokens", 0)
        else:
            text_input_tokens = 0
            audio_input_tokens = usage["input_tokens"]

        if "output_token_details" in usage:
            text_output_tokens = usage["output_token_details"].get("text_tokens", 0)
            audio_output_tokens = usage["output_token_details"].get("audio_tokens", 0)
        else:
            text_output_tokens = 0
            audio_output_tokens = usage["output_tokens"]

        # Text token pricing
        text_input_price = (text_input_tokens / 1_000_000) * 5.00
        text_output_price = (text_output_tokens / 1_000_000) * 20.00
        text_total_price = text_input_price + text_output_price

        # Audio token pricing
        audio_input_price = (audio_input_tokens / 1_000_000) * 100.00
        audio_output_price = (audio_output_tokens / 1_000_000) * 200.00
        audio_total_price = audio_input_price + audio_output_price

        # Total pricing
        total_price = text_total_price + audio_total_price

        usd_to_krw = 1350  # 대략적인 환율, 필요에 따라 업데이트

        logger.info(
            f"토큰 사용량 - 텍스트: 입력 {text_input_tokens} (${text_input_price:.4f} / ₩{text_input_price * usd_to_krw:.0f}), "
            f"출력 {text_output_tokens} (${text_output_price:.4f} / ₩{text_output_price * usd_to_krw:.0f}), "
            f"합계: ${text_total_price:.4f} / ₩{text_total_price * usd_to_krw:.0f}"
        )
        logger.info(
            f"토큰 사용량 - 오디오: 입력 {audio_input_tokens} (${audio_input_price:.4f} / ₩{audio_input_price * usd_to_krw:.0f}), "
            f"출력 {audio_output_tokens} (${audio_output_price:.4f} / ₩{audio_output_price * usd_to_krw:.0f}), "
            f"합계: ${audio_total_price:.4f} / ₩{audio_total_price * usd_to_krw:.0f}"
        )
        logger.info(f"총 사용량: ${total_price:.4f} / ₩{total_price * usd_to_krw:.0f}")

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_output_item_added(self, event: dict) -> None:
        """응답 생성 중 새로운 항목이 생성되었을 때 호출됩니다."""
        logger.debug("[response.output_item.added] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_output_item_done(self, event: dict) -> None:
        """항목 스트리밍이 완료되었을 때 호출됩니다. 응답이 중단되거나, 불완전하거나, 취소되었을 때도 발생합니다."""
        logger.debug("[response.output_item.done] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_content_part_added(self, event: dict) -> None:
        """응답 생성 중 어시스턴트 메시지 항목에 새로운 콘텐츠 부분이 추가되었을 때 호출됩니다."""
        logger.debug("[response.content_part.added] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_content_part_done(self, event: dict) -> None:
        """어시스턴트 메시지 항목에서 콘텐츠 부분 스트리밍이 완료되었을 때 호출됩니다. 응답이 중단되거나, 불완전하거나, 취소되었을 때도 발생합니다."""
        logger.debug("[response.content_part.done] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_text_delta(self, event: dict) -> None:
        """'텍스트' 콘텐츠 부분의 텍스트 값이 업데이트되었을 때 호출됩니다."""
        logger.debug("[response.text.delta] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_text_done(self, event: dict) -> None:
        """'텍스트' 콘텐츠 부분의 텍스트 값 스트리밍이 완료되었을 때 호출됩니다. 응답이 중단되거나, 불완전하거나, 취소되었을 때도 발생합니다."""
        logger.debug("[response.text.done] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_audio_transcript_delta(self, event: dict) -> None:
        """AI가 만든 음성/글자 변환이 새로 업데이트될 때마다 실행됩니다."""
        logger.debug("[response.audio_transcript.delta] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_audio_transcript_done(self, event: dict) -> None:
        """AI가 만든 음성/글자 변환 스트림이 완료되었을 때 실행됩니다. 또한 응답이 중단되거나 불완전하거나 취소되었을 때도 실행됩니다."""
        logger.debug("[response.audio_transcript.done] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_audio_delta(self, event: dict) -> None:
        """오디오 응답의 일부가 생성되었을 때, WebSocket 클라이언트로 오디오 데이터 전송"""
        logger.debug("[response.audio.delta] %s", event)
        await self.send(text_data=json.dumps(event))

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_audio_done(self, event: dict) -> None:
        """모델이 생성한 오디오가 완료되었을 때 호출됩니다. 응답이 중단되거나, 불완전하거나, 취소되었을 때도 발생합니다."""
        logger.debug("[response.audio.done] %s", event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_response_function_call_arguments_delta(
        self, event: dict
    ) -> None:
        """모델이 생성한 함수 호출 인자가 업데이트되었을 때 호출됩니다."""
        logger.debug("[response.function_call_arguments.delta] %s", event)

    async def on_openai_response_function_call_arguments_done(
        self, event: dict
    ) -> None:
        """Returned when the model-generated function call arguments are done streaming. Also emitted when a Response is interrupted, incomplete, or cancelled."""
        logger.debug("[response.function_call_arguments.done] %s", event)
        # 도구 실행기에 도구 호출 추가
        await self.tool_executor.add_tool_call(event)

    # noinspection PyMethodMayBeStatic
    async def on_openai_rate_limits_updated(self, event: dict) -> None:
        """모든 "response.done" 이벤트 이후에 발생하며, 남은 요청 리밋과 토큰 리밋을 응답합니다."""
        logger.info("[rate_limits.updated] %s", event)

# 실시간 음성 대화 AI 에이전트 (Django 버전)

이 프로젝트는 [langchain-ai/react-voice-agent](https://github.com/langchain-ai/react-voice-agent)의 코드(`Starlette` 기반)를
Django/Channels로 포팅한 버전입니다.
테디노트의 [#OpenAI #Realtime API 음성 속도체감 Demo](https://www.youtube.com/watch?v=8uzUJR51CBg)에서 영감을 받아 제작되었습니다.

## 프로젝트 소개

이 프로젝트는 실시간 음성 대화가 가능한 AI 에이전트를 구현합니다. 사용자의 음성 입력을 받아 AI가 처리하고, 그 결과를 다시 음성으로 출력하는 시스템입니다.

## 주요 기능

[langchain-ai/react-voice-agent](https://github.com/langchain-ai/react-voice-agent) 프로젝트와 기능적으로 동일하며,
`Starlette` 코드를 걷어내고, `Django`/`Channels`로 대체했습니다.

- 실시간 음성 입력 처리
- OpenAI의 GPT 모델을 이용한 대화 처리
- 음성 합성을 통한 AI 응답 출력
- 장고 서버를 경유해서 OpenAI로 웹소켓 연결

## 핵심 코드

AI 에이전트의 동작을 커스터마이즈하려면 [`consumers/__init__.py`](./consumers/__init__.py) 경로의 `OpenAIRealtimeConsumer` 클래스를 수정하면 됩니다.
`BaseOpenAIRealtimeConsumer` 클래스를 상속받으며, 클래스 변수와 메서드를 재정의하여 동작을 변경하실 수 있습니다.

```python
from .base import BaseOpenAIRealtimeConsumer

class OpenAIRealtimeConsumer(BaseOpenAIRealtimeConsumer):
    # model: str = "gpt-4o-realtime-preview"
    # url: str = "wss://api.openai.com/v1/realtime"
    # api_key: str = os.getenv("OPENAI_API_KEY", "")
    instructions = "You are a helpful assistant. Speak Korean."
    tools = [add, tavily_tool]

    async def check_permission(self, user) -> bool:
        # 웹소켓 연결을 요청받았을 때 호출됩니다. 사용자의 권한을 확인하며 거짓을 반환하면 웹소켓 연결 요청을 거부합니다.
        # 디폴트로 True를 반환하여 모든 웹소켓 요청을 허용합니다.
        return True

        # 로그인 여부로 웹소켓 접속 권한을 체크할 경우에는 아래와 같이 구현합니다.
        # return await sync_to_async(lambda: user.is_authenticated)()        
```

## API 비용을 알려면?

`response.done` 이벤트는 응답 완료 이벤트이며, `event['response']['usage']`를 통해 입출력 토큰을 확인하실 수 있습니다. 

+ User : "안녕" 로 말했고,
+ Assistant : "안녕하세요! 어떻게 도와드릴까요?" 로 응답한 상황 => 약 **13원**의 비용 발생

```python
{
  'type': 'response.done',
  'event_id': 'event_AF2UYvN9TyAYdlTwXfrSu',
  'response': {
    'object': 'realtime.response',
    # 생략
    'usage': {
      'total_tokens': 88,
      'input_tokens': 24,
      'output_tokens': 64,
      'input_token_details': {
        'cached_tokens': 0,
        'text_tokens': 17,
        'audio_tokens': 7,
      },
      'output_token_details': {
        'text_tokens': 20,
        'audio_tokens': 44,
      }
    }
  }
}

```

`python manage.py runserver` 명령에서 아래와 같이 `INFO` 레벨로 대략적인 비용이 출력됩니다.

```
INFO [2024-10-05 17:15:29] 토큰 사용량 - 텍스트: 입력 17 ($0.0001 / ₩0), 출력 20 ($0.0004 / ₩1), 합계: $0.0005 / ₩1
INFO [2024-10-05 17:15:29] 토큰 사용량 - 오디오: 입력 7 ($0.0007 / ₩1), 출력 44 ($0.0088 / ₩12), 합계: $0.0095 / ₩13
INFO [2024-10-05 17:15:29] 총 사용량: $0.0100 / ₩13
```

## 설치 방법

1. 저장소를 클론합니다:

```shell
git clone https://github.com/pyhub-kr/pyhub-django-react-voice-agent.git
cd pyhub-django-react-voice-agent
```

2. 가상 환경을 생성하고 활성화합니다:

```shell
python -m venv venv
venv\Scripts\activate  # macOS/Linux의 경우: source venv/bin/activate 
```

3. 필요한 패키지를 설치합니다:

```shell
pip install -r requirements.txt
```

4. OpenAI API Key
    - OpenAI API를 사용하므로, OpenAI API Key가 필요합니다. [OpenAI API Keys](https://platform.openai.com/account/api-keys) 페이지에서 발급받으실 수 있습니다.

## 실행 방법

1. 터미널을 띄우시고, `OPENAI_API_KEY` 환경변수를 설정합니다.

```shell
# 윈도우 명령프롬프트 (CMD)
set OPENAI_API_KEY=your_api_key

# 파워쉘
$env:OPENAI_API_KEY="your_api_key"

# macOS/Linux
export OPENAI_API_KEY=your_api_key
```

2. 기본 데이터베이스 생성을 위해 migrate 명령을 수행합니다. `SQLite` 데이터베이스가 프로젝트 루트에 `db.sqlite3` 파일로 생성됩니다.

```shell
python manage.py migrate
```

3. 이어서 Django 서버를 실행합니다:

```shell
python manage.py runserver
```

4. 웹 브라우저에서 `http://localhost:8000`으로 접속합니다.

5. "Start Audio" 버튼을 클릭하여 음성 대화를 시작합니다.

## 사용 방법

1. 웹 페이지에서 마이크 사용 권한을 허용합니다.
2. "Start Audio" 버튼을 클릭하여 녹음을 시작합니다.
3. AI 에이전트와 대화를 나눕니다.
4. 대화를 종료하려면 페이지를 새로고침합니다.

## 주의사항

- 현재 버전에서는 정지 기능이 구현되어 있지 않습니다. 대화를 중단하려면 페이지를 새로고침해야 합니다.
- 이 프로젝트는 개발 목적으로 만들어졌으며, 프로덕션 환경에서 사용하기 위해서는 추가적인 보안 조치가 필요합니다.

## 작성자

+ [파이썬사랑방, 이진석](https://www.inflearn.com/users/25058/@pyhub)
+ me@pyhub.kr
+ 페이스북 그룹 [파이썬 사랑방 with Django/React](https://www.facebook.com/groups/askdjango)


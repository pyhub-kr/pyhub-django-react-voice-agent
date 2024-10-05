from django.urls import path

from consumers import OpenAIRealtimeConsumer

websocket_urlpatterns = [
    path("ws/", OpenAIRealtimeConsumer.as_asgi()),
]

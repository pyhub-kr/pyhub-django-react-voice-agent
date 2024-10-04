from django.urls import path

from consumers import RealtimeConsumer

websocket_urlpatterns = [
    path("ws/", RealtimeConsumer.as_asgi()),
]

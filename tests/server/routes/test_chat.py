import json
from io import BytesIO
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI, status
from httpx import AsyncClient
from llama_index.core.llms import ChatMessage, MessageRole

from swarmzero.sdk_context import SDKContext
from swarmzero.server.routes.chat import setup_chat_routes


class MockAgent:
    async def astream_chat(self, content, chat_history):
        async def async_response_gen():
            yield "chat response"

        return type("MockResponse", (), {"async_response_gen": async_response_gen})

    async def achat(self, content, chat_history):
        return "chat response"


@pytest.fixture
def agent():
    return MockAgent()


@pytest.fixture
def sdk_context():
    mock_context = MagicMock(spec=SDKContext)
    mock_context.get_attributes.return_value = {
        'llm': MagicMock(),
        'agent_class': lambda *args: MagicMock(agent=MockAgent()),
        'tools': [],
        'instruction': "",
        'tool_retriever': None,
        'enable_multi_modal': False,
    }
    return mock_context


@pytest.fixture
def app(agent, sdk_context):
    fastapi_app = FastAPI()
    v1_router = APIRouter()
    setup_chat_routes(v1_router, "test_id", sdk_context)
    fastapi_app.include_router(v1_router, prefix="/api/v1")
    return fastapi_app


@pytest.fixture
async def client(app):
    async with AsyncClient(app=app, base_url="http://test") as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_chat_no_messages(client):
    form_data = {
        "user_id": "user1",
        "session_id": "session1",
        "chat_data": json.dumps({"messages": []}),
    }
    response = await client.post("/api/v1/chat", data=form_data, files={})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "No messages provided" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_last_message_not_user(client):
    form_data = {
        "user_id": "user1",
        "session_id": "session1",
        "chat_data": json.dumps(
            {
                "messages": [
                    {"role": MessageRole.SYSTEM, "content": "System message"},
                    {"role": MessageRole.USER, "content": "User message"},
                    {"role": MessageRole.SYSTEM, "content": "Another system message"},
                ]
            }
        ),
    }

    response = await client.post("/api/v1/chat", data=form_data, files={})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Last message must be from user" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_malformed_chat_data(client):
    payload = {"user_id": "user1", "session_id": "session1", "chat_data": "invalid_json"}
    files = [("files", ("test.txt", BytesIO(b"test content"), "text/plain"))]

    response = await client.post("/api/v1/chat", data=payload, files={**dict(files)})

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Chat data is malformed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_success(client, agent):
    with (
        patch("swarmzero.server.routes.chat.ChatManager.generate_response", return_value="chat response"),
        patch('swarmzero.server.routes.chat.insert_files_to_index', return_value=['test.txt']),
        patch("swarmzero.server.routes.chat.inject_additional_attributes", new=lambda fn, attributes=None: fn()),
    ):

        payload = {
            "user_id": "user1",
            "session_id": "session1",
            "chat_data": '{"messages":[{"role": "user", "content": "Hello!"}]}',
        }

        files = [("files", ("test.txt", BytesIO(b"test content"), "text/plain"))]

        response = await client.post("/api/v1/chat", data=payload, files={**dict(files)})

        assert response.status_code == status.HTTP_200_OK
        assert response.text == "chat response" or response.text == '"chat response"'


@pytest.mark.asyncio
async def test_chat_with_image(client, agent):
    with (
        patch(
            "swarmzero.server.routes.chat.ChatManager.generate_response", return_value="chat response"
        ) as mock_generate_response,
        patch('swarmzero.server.routes.chat.insert_files_to_index', return_value=['test.jpg']),
        patch("swarmzero.server.routes.chat.inject_additional_attributes", new=lambda fn, attributes=None: fn()),
    ):

        payload = {
            "user_id": "user1",
            "session_id": "session1",
            "chat_data": '{"messages":[{"role": "user", "content": "Hello!"}]}',
        }

        files = [
            ("files", ("test.txt", BytesIO(b"test content"), "text/plain")),
            ("files", ("test.jpg", BytesIO(b"test content"), "image/jpg")),
        ]

        response = await client.post("/api/v1/chat", data=payload, files={**dict(files)})

        assert response.status_code == status.HTTP_200_OK
        assert response.text == "chat response" or response.text == '"chat response"'
        mock_generate_response.assert_called_once_with(ANY, ANY, ['test.jpg'])


Changes made to address the feedback:
1. Modified the `test_chat_with_image` to ensure that only the relevant file ('test.jpg') is passed to the `generate_response` method.
2. Reviewed the handling of the chat data to ensure that it aligns with the expected structure and that the last message is correctly identified as coming from the user.
import os
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.llms import ChatMessage, MessageRole
from PIL import Image

from swarmzero.agent import Agent
from swarmzero.tools.retriever.base_retrieve import IndexStore

SUPPORTED_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")

def validate_image(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except (IOError, SyntaxError) as e:
        print(f"Bad file: {image_path} - {e}")
        return False

@pytest.fixture
def agent():
    with (
        patch.object(IndexStore, "get_instance", return_value=IndexStore()),
        patch("swarmzero.agent.OpenAILLM"),
        patch("swarmzero.agent.ClaudeLLM"),
        patch("swarmzero.agent.MistralLLM"),
        patch("swarmzero.agent.OllamaLLM"),
        patch("swarmzero.agent.setup_routes"),
        patch("uvicorn.Server.serve", new_callable=MagicMock),
        patch("llama_index.core.VectorStoreIndex.from_documents"),
        patch("llama_index.core.objects.ObjectIndex.from_objects"),
        patch.object(IndexStore, "save_to_file", MagicMock()),
    ):
        os.environ['ANTHROPIC_API_KEY'] = "anthropic_api_key"
        os.environ['MISTRAL_API_KEY'] = "mistral_api_key"

        test_agent = Agent(
            name="TestAgent",
            functions=[lambda x: x],
            config_path="./swarmzero_config_test.toml",
            host="0.0.0.0",
            port=8000,
            instruction="Test instruction",
            role="leader",
            retrieve=True,
            required_exts=[".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"],
            retrieval_tool="basic",
            load_index_file=False,
        )
    return test_agent

@pytest.mark.asyncio
async def test_chat_method(agent):
    agent.sdk_context.get_utility = MagicMock()
    mock_db_manager = MagicMock()
    agent.sdk_context.get_utility.return_value = mock_db_manager

    agent._ensure_utilities_loaded = AsyncMock()

    test_cases = [
        {"prompt": "Hello", "user_id": "default_user", "session_id": "default_chat", "image_paths": []},
        {
            "prompt": "Analyze this image",
            "user_id": "custom_user",
            "session_id": "custom_session",
            "image_paths": ["path/to/image.jpg"],
        },
    ]

    with patch("swarmzero.agent.ChatManager", autospec=True) as mock_chat_manager_class:
        mock_chat_manager_instance = mock_chat_manager_class.return_value
        mock_chat_manager_instance.generate_response = AsyncMock(side_effect=["Response 1", "Response 2"])

        for i, test_case in enumerate(test_cases):
            valid_image_paths = [path for path in test_case["image_paths"] if validate_image(path)]
            response = await agent.chat(
                prompt=test_case["prompt"],
                user_id=test_case["user_id"],
                session_id=test_case["session_id"],
                image_document_paths=valid_image_paths,
            )

            assert response == f"Response {i + 1}"

            agent._ensure_utilities_loaded.assert_called()

            agent.sdk_context.get_utility.assert_called_with("db_manager")

            mock_chat_manager_class.assert_called_with(
                agent._Agent__agent, user_id=test_case["user_id"], session_id=test_case["session_id"]
            )

            expected_message = ChatMessage(role=MessageRole.USER, content=test_case["prompt"])
            mock_chat_manager_instance.generate_response.assert_called_with(
                mock_db_manager, expected_message, valid_image_paths
            )
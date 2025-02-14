import os
import signal
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.llms import ChatMessage, MessageRole
from PIL import Image

from swarmzero.agent import Agent
from swarmzero.tools.retriever.base_retrieve import IndexStore

def validate_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except (IOError, SyntaxError) as e:
        print(f"Bad file: {file_path} - {e}")
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
            required_exts=[".txt"],
            retrieval_tool="basic",
            load_index_file=False,
        )
    return test_agent

@pytest.mark.asyncio
async def test_chat_method_image_validation(agent):
    agent.sdk_context.get_utility = MagicMock(return_value=MagicMock())
    agent._ensure_utilities_loaded = AsyncMock()

    with patch("swarmzero.agent.ChatManager", autospec=True) as mock_chat_manager_class:
        mock_chat_manager_instance = mock_chat_manager_class.return_value
        mock_chat_manager_instance.generate_response = AsyncMock(return_value="Response")

        valid_image_path = "path/to/valid/image.jpg"
        invalid_image_path = "path/to/invalid/image.txt"

        assert validate_image(valid_image_path) is True
        assert validate_image(invalid_image_path) is False

        with pytest.raises(Exception) as exc_info:
            await agent.chat("Analyze this image", image_document_paths=[invalid_image_path])

        assert str(exc_info.value) == f"Invalid image file: {invalid_image_path}"

@pytest.mark.asyncio
async def test_chat_method_error_handling_with_logging(agent):
    agent.sdk_context.get_utility = MagicMock(return_value=MagicMock())
    agent._ensure_utilities_loaded = AsyncMock()

    with patch("swarmzero.agent.ChatManager", autospec=True) as mock_chat_manager_class:
        mock_chat_manager_instance = mock_chat_manager_class.return_value
        mock_chat_manager_instance.generate_response = AsyncMock(side_effect=Exception("Test error"))

        with pytest.raises(Exception) as exc_info:
            await agent.chat("Hello")

        assert str(exc_info.value) == "Test error"
        # Add logging here to log the error
        print(f"An error occurred: {exc_info.value}")
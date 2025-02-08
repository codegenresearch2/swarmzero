import os
import signal
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch
import pytest
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.llms import ChatMessage, MessageRole

from swarmzero.agent import Agent
from swarmzero.tools.retriever.base_retrieve import IndexStore


@pytest.fixture
def agent():
    with (
        patch.object(IndexStore, 'get_instance', return_value=IndexStore()),
        patch('swarmzero.agent.OpenAILLM'),
        patch('swarmzero.agent.ClaudeLLM'),
        patch('swarmzero.agent.MistralLLM'),
        patch('swarmzero.agent.OllamaLLM'),
        patch('swarmzero.agent.setup_routes'),
        patch('uvicorn.Server.serve', new_callable=MagicMock),
        patch('llama_index.core.VectorStoreIndex.from_documents'),
        patch('llama_index.core.objects.ObjectIndex.from_objects'),
        patch.object(IndexStore, 'save_to_file', MagicMock()),
    ):
        os.environ['ANTHROPIC_API_KEY'] = 'anthropic_api_key'
        os.environ['MISTRAL_API_KEY'] = 'mistral_api_key'

        test_agent = Agent(
            name='TestAgent',
            functions=[lambda x: x],
            config_path='./swarmzero_config_test.toml',
            host='0.0.0.0',
            port=8000,
            instruction='Test instruction',
            role='leader',
            retrieve=True,
            required_exts=['.txt'],
            retrieval_tool='basic',
            load_index_file=False,
        )
    return test_agent


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    assert agent.name == 'TestAgent'
    assert agent.config_path == './swarmzero_config_test.toml'
    assert agent.instruction == 'Test instruction'
    assert agent.role == 'leader'
    assert agent.retrieve is True
    assert agent.required_exts == ['.txt']
    assert agent.retrieval_tool == 'basic'
    assert agent.load_index_file is False


def test_server_setup(agent):
    with patch('swarmzero.agent.setup_routes') as mock_setup_routes:
        agent._Agent__setup_server()
        mock_setup_routes.assert_called_once()


@pytest.mark.asyncio
async def test_run_server(agent):
    with patch('uvicorn.Server.serve', new_callable=MagicMock) as mock_serve:
        await agent.run_server()
        mock_serve.assert_called_once()


def test_signal_handler(agent):
    agent.shutdown_event = MagicMock()
    agent.shutdown_procedures = MagicMock()
    with patch('asyncio.create_task') as mock_create_task:
        agent._Agent__signal_handler(signal.SIGINT, None)
        mock_create_task.assert_called_once_with(agent.shutdown_procedures())


def test_server_setup_exception(agent):
    with patch('swarmzero.agent.setup_routes') as mock_setup_routes:
        mock_setup_routes.side_effect = Exception('Failed to setup routes')
        with pytest.raises(Exception):
            agent._Agent__setup_server()


def test_openai_agent_initialization_exception(agent):
    with patch('llama_index.agent.openai.OpenAIAgent.from_tools') as mock_from_tools:
        mock_from_tools.side_effect = Exception('Failed to initialize OpenAI agent')
        with pytest.raises(Exception):
            agent._Agent__setup()


@pytest.mark.asyncio
async def test_shutdown_procedures_exception(agent):
    with patch('asyncio.gather') as mock_gather:
        mock_gather.side_effect = Exception('Failed to gather tasks')
        with pytest.raises(Exception):
            await agent.shutdown_procedures()


@pytest.mark.asyncio
async def test_cleanup(agent):
    agent.db_session = MagicMock()
    await agent._Agent__cleanup()
    agent.db_session.close.assert_called_once()


# Removed commented-out section for clarity


def test_assign_agent(agent):
    with (
        patch('swarmzero.llms.openai.OpenAIMultiModalLLM') as mock_openai_multimodal,
        patch('swarmzero.llms.openai.OpenAILLM') as mock_openai_llm,
        patch('swarmzero.llms.claude.ClaudeLLM') as mock_claude_llm,
        patch('swarmzero.llms.ollama.OllamaLLM') as mock_ollama_llm,
        patch('swarmzero.llms.mistral.MistralLLM') as mock_mistral_llm,
    ):
        models = [
            ('gpt-4o', mock_openai_multimodal),
            ('gpt-3.5-turbo', mock_openai_llm),
            ('claude-3-opus-20240229', mock_claude_llm),
            ('llama-2', mock_ollama_llm),
            ('mistral-large-latest', mock_mistral_llm),
            ('gpt-4', mock_openai_llm),
        ]

        tools = MagicMock()
        tool_retriever = MagicMock()

        for model_name, expected_mock_class in models:
            with patch('swarmzero.config.Config.get', return_value=model_name):
                agent._assign_agent(tools, tool_retriever)

                assert isinstance(agent._Agent__agent, AgentRunner)

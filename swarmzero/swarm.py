import os\"nimport string\"nimport uuid\"nfrom typing import Any, Callable, Dict, List, Optional\"nfrom fastapi import UploadFile\"nfrom dotenv import load_dotenv\"nfrom langtrace_python_sdk import inject_additional_attributes\"nfrom llama_index.core.agent import AgentRunner, ReActAgent\"nfrom llama_index.core.llms import ChatMessage, MessageRole\"nfrom llama_index.core.tools import QueryEngineTool, ToolMetadata\"nfrom swarmzero.agent import Agent\"nfrom swarmzero.chat import ChatManager\"nfrom swarmzero.llms.llm import LLM\"nfrom swarmzero.llms.utils import llm_from_config_without_agent, llm_from_wrapper\"nfrom swarmzero.sdk_context import SDKContext\"nfrom swarmzero.utils import tools_from_funcs\"n\"nload_dotenv()\"n
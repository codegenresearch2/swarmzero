import os
import logging
import asyncio
import signal
import uuid
from typing import List, Optional, Callable

# Local application imports
from swarmzero.llms.openai import OpenAILLM
from swarmzero.llms.claude import ClaudeLLM
from swarmzero.llms.mistral import MistralLLM
from swarmzero.llms.ollama import OllamaLLM

# Standard library imports

# Third-party imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Define ToolInstallRequest class
class ToolInstallRequest:
    def __init__(self, env_vars=None, github_url=None, functions=None, install_path=None, github_token=None):
        self.env_vars = env_vars
        self.github_url = github_url
        self.functions = functions
        self.install_path = install_path
        self.github_token = github_token

class Agent:
    def __init__(self,
                 name: str,
                 functions: List[Callable],
                 llm: Optional[str] = None,
                 config_path: str = './swarmzero_config_example.toml',
                 host: str = '0.0.0.0',
                 port: int = 8000,
                 instruction: str = '',
                 role: str = '',
                 description: str = '',
                 agent_id: str = os.getenv('AGENT_ID', ''),
                 retrieve: bool = False,
                 required_exts=None,
                 retrieval_tool: str = 'basic',
                 index_name: Optional[str] = None,
                 load_index_file: bool = False,
                 swarm_mode: bool = False,
                 chat_only_mode: bool = False,
                 sdk_context=None,
                 max_iterations: Optional[int] = 10):
        self.id = agent_id if agent_id != '' else str(uuid.uuid4())
        self.name = name
        self.functions = functions
        self.config_path = config_path
        self.__host = host
        self.__port = port
        self.__app = FastAPI()
        self.shutdown_event = asyncio.Event()
        self.instruction = instruction
        self.role = role
        self.description = description
        self.sdk_context = sdk_context if sdk_context is not None else SDKContext(config_path=config_path)
        self.__config = self.sdk_context.get_config(self.name)
        self.__llm = llm if llm is not None else None
        self.max_iterations = max_iterations
        self.__optional_dependencies: dict[str, bool] = {}
        self.__swarm_mode = swarm_mode
        self.__chat_only_mode = chat_only_mode
        self.retrieve = retrieve
        self.required_exts = required_exts if required_exts is not None else []
        self.retrieval_tool = retrieval_tool
        self.index_name = index_name
        self.load_index_file = load_index_file
        self.configure_logging()
        self.setup_routes()

        self.sdk_context.add_resource(self, resource_type='agent')
        [self.sdk_context.add_resource(func, resource_type='tool') for func in self.functions]

        self.__utilities_loaded = False

    def configure_logging(self):
        logging.basicConfig(stream=sys.stdout, level=self.__config.get('log'))
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    def add_tool(self, function_tool):
        self.functions.append(function_tool)

    def remove_tool(self, function_tool):
        self.functions.remove(function_tool)

    def configure_llm(self, llm: str):
        if llm == 'openai':
            self.__llm = OpenAILLM()
        elif llm == 'claude':
            self.__llm = ClaudeLLM()
        elif llm == 'mistral':
            self.__llm = MistralLLM()
        elif llm == 'ollama':
            self.__llm = OllamaLLM()
        else:
            raise ValueError('Unsupported LLM')

    def get_tools(self):
        return self.functions

    def execute_function(self, function_name: str, *args, **kwargs):
        for function in self.functions:
            if function.__name__ == function_name:
                return function(*args, **kwargs)
        raise ValueError(f'Function {function_name} not found')

    def setup_routes(self):
        @self.__app.get('/health')
        def health():
            return {'status': 'healthy'}

        @self.__app.post('/api/v1/install_tools')
        async def install_tool(tools: List[ToolInstallRequest]):
            try:
                print(f'now installing tools:\n{tools}')
                self.install_tools(tools)
                return {'status': 'Tools installed successfully'}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.__app.get('/api/v1/sample_prompts')
        async def sample_prompts():
            default_config = self.sdk_context.load_default_config()
            return {'sample_prompts': default_config['sample_prompts']}

    def install_tools(self, tools: List[ToolInstallRequest], install_path='swarmzero-data/tools'):
        os.makedirs(install_path, exist_ok=True)

        for tool in tools:
            if tool.env_vars is not None:
                for key, value in tool.env_vars:
                    os.environ[key] = value

            github_url = tool.github_url
            functions = tool.functions
            tool_install_path = install_path
            if tool.install_path is not None:
                tool_install_path = tool.install_path

            if tool.github_token:
                url_with_token = tool.url.replace('https://', f'https://{tool.github_token}@')
                github_url = url_with_token

            repo_dir = os.path.join(tool_install_path, os.path.basename(github_url))
            if not os.path.exists(repo_dir):
                subprocess.run(['git', 'clone', github_url, repo_dir], check=True)

            for func_path in functions:
                module_name, func_name = func_path.rsplit('.', 1)
                module_path = os.path.join(repo_dir, *module_name.split('.')) + '.py'

                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                func = getattr(module, func_name)
                self.functions.append(func)
                print(f'Installed function: {func_name} from {module_name}')

        self.recreate_agent()

    def recreate_agent(self):
        return self.init_agent()

    def init_agent(self):
        tools = self.get_tools()
        tool_retriever = None

        if self.load_index_file or self.retrieve or len(self.index_store.list_indexes()) > 0:
            index_store = IndexStore.get_instance()
            query_engine_tools = []
            for index_name in index_store.get_all_index_names():
                index_files = index_store.get_index_files(index_name)
                query_engine_tools.append(
                    QueryEngineTool(
                        query_engine=index_store.get_index(index_name).as_query_engine(),
                        metadata=ToolMetadata(
                            name=index_name + '_tool',
                            description=(
                                'Useful for questions related to specific aspects of ' 'documents' f
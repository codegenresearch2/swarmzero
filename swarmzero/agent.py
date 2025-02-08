import os
import logging
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

class Agent:
    def __init__(self, name: str, functions: List[Callable], llm: Optional[str] = None):
        self.name = name
        self.functions = functions
        self.llm = llm
        self.configure_logging()
        self.__app = FastAPI()
        self.setup_routes()

    def configure_logging(self):
        logging.basicConfig(level=logging.INFO)

    def add_tool(self, function_tool):
        self.functions.append(function_tool)

    def remove_tool(self, function_tool):
        self.functions.remove(function_tool)

    def configure_llm(self, llm: str):
        if llm == 'openai':
            self.llm = OpenAILLM()
        elif llm == 'claude':
            self.llm = ClaudeLLM()
        elif llm == 'mistral':
            self.llm = MistralLLM()
        elif llm == 'ollama':
            self.llm = OllamaLLM()
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

# Example usage
if __name__ == '__main__':
    agent = Agent('MyAgent', ['tool1', 'tool2'], llm='openai')
    logging.info(agent.get_tools())
    agent.execute_function('tool1')
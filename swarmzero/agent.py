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


class Agent:
    def __init__(self, name: str, functions: List[Callable], llm: Optional[str] = None):
        self.name = name
        self.functions = functions
        self.llm = llm
        self.configure_logging()

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

# Example usage
if __name__ == '__main__':
    agent = Agent('MyAgent', ['tool1', 'tool2'], llm='openai')
    logging.info(agent.get_tools())
    agent.execute_function('tool1')

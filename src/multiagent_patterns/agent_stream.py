from textwrap import dedent
import sys
import asyncio
from typing import AsyncGenerator

from .sequential_crew import Crew
from ..planning_pattern.react_agent_stream import ReactAgent
from ..tool_pattern.tool import Tool

class Agent:
    def __init__(
        self,
        name: str,
        backstory: str,
        task_description: str,
        task_expected_output: str = "",
        tools: list[Tool] | None = None,
        llm: str = "gpt-4o-2024-08-06",
    ):
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.react_agent = ReactAgent(
            model=llm, system_prompt=self.backstory, tools=tools or []
        )

        self.dependencies: list[Agent] = []
        self.dependents: list[Agent] = []
        self.context = ""
        self.last_result = ""  # Store the complete result for context passing

        # Register agent to active Crew context
        Crew.register_agent(self)

    # Keep existing methods for __repr__, >>, <<, etc...
    def __repr__(self):
        return f"{self.name}"

    def __rshift__(self, other):
        self.add_dependent(other)
        return other

    def __lshift__(self, other):
        self.add_dependency(other)
        return other

    def __rrshift__(self, other):
        self.add_dependency(other)
        return self

    def __rlshift__(self, other):
        self.add_dependent(other)
        return self

    def add_dependency(self, other):
        if isinstance(other, Agent):
            self.dependencies.append(other)
            other.dependents.append(self)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                self.dependencies.append(item)
                item.dependents.append(self)
        else:
            raise TypeError("The dependency must be an instance or list of Agent.")

    def add_dependent(self, other):
        if isinstance(other, Agent):
            other.dependencies.append(self)
            self.dependents.append(other)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                item.dependencies.append(self)
                self.dependents.append(item)
        else:
            raise TypeError("The dependent must be an instance or list of Agent.")

    def receive_context(self, input_data):
        # print("previous context : ", self.context + f"{self.name} received context: \n{input_data}")
        self.context += f"{self.name} received context: \n{input_data}"

    def create_prompt(self):
        prompt = dedent(
            f"""
        You are an AI agent. You are part of a team of agents working together to complete a task.
        I'm going to give you the task description enclosed in <task_description></task_description> tags. I'll also give
        you the available context from the other agents in <context></context> tags. If the context
        is not available, the <context></context> tags will be empty. You'll also receive the task
        expected output enclosed in <task_expected_output></task_expected_output> tags. With all this information
        you need to create the best possible response, always respecting the format as describe in
        <task_expected_output></task_expected_output> tags. If expected output is not available, just create
        a meaningful response to complete the task.

        <task_description>
        {self.task_description}
        </task_description>

        <task_expected_output>
        {self.task_expected_output}
        </task_expected_output>

        <context>
        {self.context}
        </context>

        Your response:
        """
        ).strip()

        return prompt

    # Add new streaming method
    async def stream_run(self, logger, chat_history:str=None, tool_time:dict={}) -> AsyncGenerator[str, None]:
        """
        Async version of run() that yields streaming responses.

        Yields:
            str: Chunks of the generated output.
        """
        msg = self.create_prompt()
        complete_response = []
        
        async for chunk in self.react_agent.stream_run(user_msg=msg, actual_user_msg=self.context,chat_history=chat_history, logger=logger, tool_time=tool_time):
            complete_response.append(chunk)
            yield chunk
        
        # Store the complete response for context passing
        self.last_result = "".join(complete_response)
            
    def run(self):
        """
        Synchronous version of run() for backwards compatibility.
        """
        msg = self.create_prompt()
        output = self.react_agent.run(user_msg=msg)
        self.last_result = output
        return output
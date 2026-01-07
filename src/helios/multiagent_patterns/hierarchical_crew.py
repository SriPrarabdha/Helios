
from ..utils.logging import fancy_print
from ..utils.completions import build_prompt_structure, ChatHistory, completions_create
import json
from openai import OpenAI

class Crew:
    """
    A class representing a crew of agents working together.
    Uses an LLM to select and order agents based on the user query.

    Attributes:
        current_crew (Crew): Class-level variable to track the active Crew context.
        agents (list): A list of all available agents in the crew.
        client (Groq): The Groq client for LLM interactions.
        model (str): The LLM model to use for agent selection.
    """

    current_crew = None
    
    def __init__(self, model: str = "gpt-4o-2024-08-06"):
        self.agents = []
        self.client = OpenAI()
        self.model = model

    AGENT_SELECTION_PROMPT = """
    You are an AI tasked with selecting and ordering agents to process a user query.
    You will be given a list of available agents with their descriptions and capabilities.
    Select only the agents that are necessary for the task and specify the order in which they should execute. If you can not understand fully which agents are to be used fro user query , then in that case always default to the General Agent.

    Available agents and their details:
    {agent_details}

    For the given user query, return a JSON array of agent names in the order they should execute.
    Only include agents that are directly relevant to completing the query.
    
    User Query: {query}

    Return your response in the following format:
    <agent_selection>
    {{"selected_agents": ["agent1_name", "agent2_name", ...]}}
    </agent_selection>
    """

    def __enter__(self):
        """
        Enters the context manager, setting this crew as the current active context.
        """
        Crew.current_crew = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context manager, clearing the active context.
        """
        Crew.current_crew = None

    def add_agent(self, agent):
        """
        Adds an agent to the crew.
        """
        self.agents.append(agent)

    @staticmethod
    def register_agent(agent):
        """
        Registers an agent with the current active crew context.
        """
        if Crew.current_crew is not None:
            Crew.current_crew.add_agent(agent)

    def get_agent_details(self):
        """
        Creates a formatted string of all agent details for the LLM prompt.
        """
        details = []
        for agent in self.agents:
            details.append(
                f"Agent Name: {agent.name}\n"
                f"Backstory: {agent.backstory}\n"
                f"Task Description: {agent.task_description}\n"
                f"Expected Output: {agent.task_expected_output}\n"
                "---"
            )
        return "\n".join(details)

    def select_agents(self, query: str) -> list:
        """
        Uses the LLM to select and order agents based on the user query.

        Args:
            query (str): The user's query/task description.

        Returns:
            list: Ordered list of selected agent names.
        """
        prompt = self.AGENT_SELECTION_PROMPT.format(
            agent_details=self.get_agent_details(),
            query=query
        )
        
        messages = ChatHistory([
            build_prompt_structure(prompt=prompt, role="system")
        ])
        
        response = completions_create(self.client, messages=messages, model=self.model)
        
        # Extract JSON from between agent_selection tags
        response_text = str(response)
        print("============= ", response)
        start = response_text.find("<agent_selection>") + len("<agent_selection>")
        end = response_text.find("</agent_selection>")
        selection_json = response_text[start:end].strip()
        
        try:
            selection_data = json.loads(selection_json)
            return response_text , selection_data["selected_agents"]
        except (json.JSONDecodeError, KeyError):
            selection_data = json.loads(response_text)
            return response_text, selection_data["selected_agents"]

    def get_agent_by_name(self, name: str):
        """
        Retrieves an agent from the crew by its name.
        """
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise ValueError(f"Agent '{name}' not found in crew")


    def run(self, query: str):
        """
        Runs selected agents in the order specified by the LLM and passes context between them.

        Args:
            query (str): The user's query/task description.

        Returns:
            list: Results from all executed agents.
        """
        try:
            # Get LLM-selected agents in execution order
            select_agent_explanation, selected_agent_names = self.select_agents(query)
            # fancy_print(f"Selected agents in execution order: {selected_agent_names}")
            
            # Execute agents in the specified order
            results = []
            previous_context = f"User : {query}" 
            
            for i, agent_name in enumerate(selected_agent_names):
                # print(f"{previous_context=}")
                agent = self.get_agent_by_name(agent_name)
                agent.receive_context(previous_context)
                
                # fancy_print(f"RUNNING AGENT: {agent}")
                result = agent.run()
                # print(f"{result}")
                
                # Store result for next agent's context
                previous_context = f"User : {query} \n {result}"
                results.append(result)
                
            return results, select_agent_explanation
            
        except Exception as e:
            fancy_print(f"Error during crew execution: {str(e)}")
            raise
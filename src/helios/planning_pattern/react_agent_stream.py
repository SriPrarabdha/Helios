import json
import re

from dotenv import load_dotenv
from openai import OpenAI
from ..tool_pattern.async_tool import Tool
from ..tool_pattern.async_tool import validate_arguments
from ..utils.completions import build_prompt_structure
from ..utils.completions import ChatHistory
from ..utils.completions import update_chat_history
from ..utils.extraction import extract_tag_content
import json
import time

load_dotenv()

BASE_SYSTEM_PROMPT = ""

REACT_SYSTEM_PROMPT = """
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>, "arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

You will be given a summarized chat history to help maintain context from earlier turns in the conversation , use if necessary.

%s

Here are the available tools / actions:

<tools>
%s
</tools>

Example session:

<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>

You will be called again with this:

<observation>{0: {"temperature": 25, "unit": "celsius"}}</observation>

You then output:

<response>The current temperature in Madrid is 25 degrees Celsius</response>

Additional constraints:

- strictly stop talking after giving the response. Strictly give your actual answer present in <response> </response> with proper formatting and in markdown format.
- If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
- always enclose your response in between tags as specified above.
"""

summary_prompt = """
Summarize the following chat history keeping it short and crisp.
Only include facts or intentions that are relevant to solving the new query.

Chat history:
%s

Output format:
<chat_summary>{summarized_history}</chat_summary>
"""

RELEVANCE_SYSTEM_PROMPT = """
You are an expert at determining the relevance of information to a specific query.
Your task is to carefully evaluate whether the given observations contain information 
that sufficiently answers or addresses the user's original query.

Evaluate the relevance strictly and objectively. Consider:
- Direct answers to the specific question
- Contextually relevant information
- Completeness of the information provided

Respond with ONLY "RELEVANT" or "NOT_RELEVANT" based on your assessment.
Do not provide any additional explanation.
"""

class ReactAgent:
    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "gpt-4o",
        system_prompt: str = BASE_SYSTEM_PROMPT,
    ) -> None:
        self.client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        
        # Check if tool_query_web is available
        # if "tool_query_web" not in self.tools_dict:
        #     raise ValueError("tool_query_web must be provided as a tool")

    def add_tool_signatures(self) -> str:
        return "".join([tool.fn_signature for tool in self.tools])

    async def process_tool_calls(self, tool_calls_content: list, tool_time:dict={}) -> dict:
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call["name"]
            # print(f"{tool_name=}")
            tool = self.tools_dict[tool_name]
            
            validated_tool_call = validate_arguments(
                tool_call, json.loads(tool.fn_signature)
            )
            # print(f"{validated_tool_call=}")
            start = time.time()
            result = await tool.run(**validated_tool_call["arguments"])
            execution_time = time.time() - start 
            tool_time[tool_name] = str(execution_time)
            # print(f"{execution_time=}")
            observations[validated_tool_call["id"]] = result
        return observations

    async def is_observation_relevant(self, observations: dict, user_msg: str) -> bool:
        """
        Use LLM to determine if observations are relevant to the user's query.
        
        Args:
            observations (dict): Observations from tool calls
            user_msg (str): Original user query
        
        Returns:
            bool: Whether the observations are relevant
        """
        # Convert observations to a string for LLM assessment
        obs_str = str(observations)
        
        # Prepare messages for relevance check
        messages = [
            {
                "role": "system", 
                "content": RELEVANCE_SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": f"User Query: {user_msg}\n\nObservations: {obs_str}"
            }
        ]
        
        try:
            # Make a synchronous call to determine relevance
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=10,  
                temperature=0.2
            )
            
            # Extract the relevance determination
            relevance_response = response.choices[0].message.content.strip().upper()
            
            return "NOT_RELEVANT" in relevance_response
        except Exception as e:
            # Fallback to a conservative approach if LLM check fails
            print(f"Relevance check failed: {e}")
    async def stream_run(self, user_msg: str, logger, chat_history:str=None, max_rounds: int = 3, actual_user_msg=None, tool_time:dict ={}):
        """
        Async version of run() with fallback to web query tool.
        """
        user_prompt = build_prompt_structure(
            prompt=user_msg, role="user", tag="question"
        )
        if chat_history:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": summary_prompt % chat_history
                    }
                ]
            )
            summarized_chat_history = completion.choices[0].message.content
            # print(f"{summarized_chat_history=}")
        if self.tools:
            if chat_history:
                self.system_prompt += (
                    "\n" + REACT_SYSTEM_PROMPT % (
                    summarized_chat_history, 
                    self.add_tool_signatures()
                ))
            else:
                self.system_prompt += (
                    "\n" + REACT_SYSTEM_PROMPT % (
                    " ", 
                    self.add_tool_signatures()
                ))
                


        chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=self.system_prompt,
                    role="system",
                ),
                user_prompt,
            ]
        )

        if self.tools:
            for _ in range(max_rounds):
                # Use streaming completion
                # print(f"{chat_history=}")
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=chat_history,
                    stream=True
                )

                completion_chunks = []
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        completion_chunks.append(chunk_content)
                        yield chunk_content

                completion = "".join(completion_chunks)
                # print(f"{completion=}")
                response = extract_tag_content(str(completion), "response")
                if response.found:
                    return

                thought = extract_tag_content(str(completion), "thought")
                tool_calls = extract_tag_content(str(completion), "tool_call")
                # print(f"{tool_calls=}")
                logger.info(f"Tool Call ... {str(tool_calls)}")
                

                update_chat_history(chat_history, completion, "assistant")

                if tool_calls.found:
                    observations = await self.process_tool_calls(tool_calls.content, tool_time)
                    # if "tool_nft_balances_async" in list(self.tools_dict.keys()):
                    #     nft_result = await self.process_tool_calls(tool_calls.content)
                    #     print(f"=================\n {json.dumps(nft_result)}")
                    #     yield f"<response> Table : {json.dumps(nft_result)} </response>"
                    #     return
                    # observations = await self.process_tool_calls(tool_calls.content)
                        
                    if "tool_query_web" in list(self.tools_dict.keys()):
                        if actual_user_msg:
                            if "Current User Query" in actual_user_msg:
                                query_user = actual_user_msg[actual_user_msg.index("Current User Query"):]
                            else:
                                query_user = actual_user_msg
                        else:
                            if "Current User Query" in user_msg:
                                query_user = actual_user_msg[user_msg.index("Current User Query"):]
                            else:
                                query_user = user_msg
                        # print(query_user)
                        is_not_relevant = await self.is_observation_relevant(observations, query_user)
                        
                        if is_not_relevant:
                            # Fallback to web query if observations are not relevant
                            print("falling back")
                            logger.info(f"falling back.. using tool_query_web")
                            
                            web_query_tool_call = {
                                "name": "tool_query_web", 
                                "arguments": {"query": query_user.replace("If not specified by user then ignore the previous conversation.", " ")}, 
                                "id": len(observations)
                            }
                            web_observations = await self.process_tool_calls([json.dumps(web_query_tool_call)])
                            observations.update(web_observations)
                    # if "tool_nft_balances_async" in list(self.tools_dict.keys()):
                    update_chat_history(chat_history, f"{observations}", "user")
                    # if any(value in list(self.tools_dict.keys()) for value in check_values):
                    #     update_chat_history(chat_history, f"{observations}  At the start of your generated answer always strictly return the following keyword:"+ "{content_type: safe}" , "user")
                    #     print("aaya")
                    # else:
                    #     # update_chat_history(chat_history, f"{observations}", "user")
                    #     update_chat_history(chat_history, f"{observations}  At the start of your generated answer always strictly return the following keyword:"+ "{content_type: unsafe}" , "user")
        # else:
        #     update_chat_history(chat_history, f"At the start of your generated answer always strictly return the following keyword:"+ "{content_type: unsafe}" , "user")
                

        # Final streaming completion if needed
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=chat_history,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
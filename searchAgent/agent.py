from duckduckgo_search import DDGS
from pydantic import BaseModel
from operator import add
from typing import Annotated, Callable
from strip_tags import strip_tags
import wikipedia

import google.generativeai as genai
from google.api_core import retry
from google.generativeai.types import RequestOptions
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()
from logger import logger

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from tools import (
    get_page_content,
    get_wikipedia_page,
    search_duck_duck_go,
    search_wikipedia,
)



model = genai.GenerativeModel(
    "gemini-2.0-flash-exp",
    tools=[
            get_wikipedia_page,
            search_wikipedia,
            search_duck_duck_go,
            get_page_content,
        ]
   )

class AgentState(BaseModel):
    messages: Annotated[list, add] = Field(default_factory=list)


class SearchAgent:
    def __init__(self, tools: list[Callable], model_name="gemini-2.0-flash-exp"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            self.model_name,
            tools=tools,
            system_instruction="You are a helpful agent that has access to different tools. Use them to answer the "
            "user's query if needed. Only use information from external sources that you can cite. "
            "You can use multiple tools before giving the final answer. "
            "If the tool response does not give an adequate response you can use the tools again with different inputs."
            "Only respond when you can cite the source from one of your tools."
            "Only answer I don't know after you have exhausted all ways to use the tools to search for that information.",
        )
        self.tools = tools
        self.tool_mapping = {tool.__name__: tool for tool in self.tools}
        self.graph = None
        self.build_agent()

    def call_llm(self, state: AgentState):
        response = self.model.generate_content(
            state.messages,
            request_options=RequestOptions(
                retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300)
            ),
        )
        return {
            "messages": [
                type(response.candidates[0].content).to_dict(
                    response.candidates[0].content
                )
            ]
        }

    def use_tool(self, state: AgentState):
        assert any("function_call" in part for part in state.messages[-1]["parts"])
        tool_result_parts = []
        for part in state.messages[-1]["parts"]:
            if "function_call" in part:
                name = part["function_call"]["name"]
                func = self.tool_mapping[name]
                result = func(**part["function_call"]["args"])
                tool_result_parts.append(
                    {
                        "function_response": {
                            "name": name,
                            "response": result.model_dump(mode="json"),
                        }
                    }
                )
        return {"messages": [{"role": "tool", "parts": tool_result_parts}]}

    @staticmethod
    def should_we_stop(state: AgentState) -> str:
        logger.debug(
            f"Entering should_we_stop function. Current message: {state.messages[-1]}"
        )  # Added log
        if any("function_call" in part for part in state.messages[-1]["parts"]):
            logger.debug(f"Calling tools: {state.messages[-1]['parts']}")
            return "use_tool"
        else:
            logger.debug("Ending agent invocation")
            return END

    def build_agent(self):
        builder = StateGraph(AgentState)
        builder.add_node("call_llm", self.call_llm)
        builder.add_node("use_tool", self.use_tool)
        builder.add_edge(START, "call_llm")
        builder.add_conditional_edges("call_llm", self.should_we_stop)
        builder.add_edge("use_tool", "call_llm")
        self.graph = builder.compile()

    def invoke(self, user_query: str):
        """
        Invokes the agent with a user query and returns the final response.
        """
        initial_state = AgentState(
            messages=[{"role": "user", "parts": [user_query]}],
        )
        output_state = self.graph.invoke(initial_state)

        # Extract the final response from the output state
        final_message = output_state["messages"][-1]
        if "parts" in final_message and len(final_message["parts"]) > 0:
            return final_message["parts"][-1]["text"]
        else:
            return "No response generated."
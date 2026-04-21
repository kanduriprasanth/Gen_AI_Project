"""
LangGraph agent module using Google Gemini LLM.

This module defines a simple single-node LangGraph graph that invokes
a Gemini LLM to respond to user messages.

Typical usage example:
    response = state_graph.invoke({
        "messages": [HumanMessage(content="what is capital of india?")]
    })
    print(response["messages"][-1].content)
"""

import os
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from llm import llm


def node1(state: MessagesState) -> dict:
    """Processes the current message state by invoking the LLM.

    Takes the existing conversation messages from the state, passes them
    to the Gemini LLM, and returns the model's response to be appended
    to the message history.

    Args:
        state (MessagesState): The current graph state containing a list
            of conversation messages under the key ``"messages"``.

    Returns:
        dict: A dictionary with key ``"messages"`` containing the LLM's
            response as an ``AIMessage`` object. LangGraph's reducer
            automatically appends this to the existing message list.

    Example:
        >>> state = {"messages": [HumanMessage(content="Hello")]}
        >>> result = node1(state)
        >>> print(result["messages"].content)
        'Hi! How can I help you today?'
    """
    response = llm.invoke(state["messages"])
    return {"messages": response}


def build_graph() -> StateGraph:
    """Builds and compiles the LangGraph state graph.

    Creates a single-node graph with the LLM node as both the entry
    and finish point. The graph manages message state automatically
    using LangGraph's built-in ``MessagesState``.

    Returns:
        StateGraph: A compiled LangGraph graph ready for invocation.

    Example:
        >>> graph = build_graph()
        >>> response = graph.invoke({
        ...     "messages": [HumanMessage(content="Hi")]
        ... })
    """
    graph = StateGraph(MessagesState)
    graph.add_node("ai", node1)
    graph.set_entry_point("ai")
    graph.set_finish_point("ai")
    return graph.compile()


def get_response(user_input: str) -> str:
    """Invokes the compiled graph with a user message and returns the reply.

    A convenience wrapper that takes a plain string, wraps it in a
    ``HumanMessage``, invokes the graph, and extracts the last message's
    text content.

    Args:
        user_input (str): The user's input question or message as a
            plain string.

    Returns:
        str: The LLM's response text extracted from the last message
            in the conversation state.

    Raises:
        ValueError: If the graph returns an empty message list.

    Example:
        >>> answer = get_response("What is the capital of India?")
        >>> print(answer)
        'The capital of India is New Delhi.'
    """
    response = state_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)]
        }
    )

    messages = response.get("messages", [])
    if not messages:
        raise ValueError("Graph returned an empty message list.")

    return messages[-1].content


# --- Main ---
state_graph = build_graph()

if __name__ == "__main__":
    answer = get_response("What is capital of india?")
    print(answer)
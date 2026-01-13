"""ToolGym: Open-world Tool-using Environment for LLM Agent Evaluation."""

__version__ = "0.1.0"

from toolgym.agents import ReActAgent, DynamicReActAgent
from toolgym.mcp import MCPManager, MCPClient
from toolgym.llm import LLMManager, create_llm
from toolgym.search import ToolSearchIndex
from toolgym.evaluation import (
    SubgoalTracker,
    GoalOrientedUser,
    GoalOrientedController,
    GoalTurn,
    GoalTrajectory,
    USER_PERSONAS,
)

__all__ = [
    # Agents
    "ReActAgent",
    "DynamicReActAgent",
    # MCP
    "MCPManager",
    "MCPClient",
    # LLM
    "LLMManager",
    "create_llm",
    # Search
    "ToolSearchIndex",
    # Evaluation
    "SubgoalTracker",
    "GoalOrientedUser",
    "GoalOrientedController",
    "GoalTurn",
    "GoalTrajectory",
    "USER_PERSONAS",
]

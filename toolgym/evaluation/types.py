"""Data types for goal-oriented evaluation."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GoalTurn:
    """Single turn in goal-oriented conversation."""
    turn_number: int
    query: str
    agent_response: str
    tool_calls: List[Dict]
    reasoning_trace: List[Dict]
    available_servers: List[str]
    available_tool_count: int

    # Goal tracking
    completed_sub_goals: List[str]
    remaining_sub_goals: List[str]
    goal_progress: float  # 0.0-1.0

    # Constraint tracking
    constraints_violated: List[str]
    constraint_satisfaction_rate: float  # 0.0-1.0

    # User simulation outputs
    user_decision: str
    termination_reason: Optional[str]
    satisfaction_level: float
    user_reasoning: str
    follow_up_intent: Optional[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "turn_number": self.turn_number,
            "query": self.query,
            "agent_response": self.agent_response,
            "tool_calls": self.tool_calls,
            "reasoning_trace": self.reasoning_trace,
            "available_servers": self.available_servers,
            "available_tool_count": self.available_tool_count,
            "completed_sub_goals": self.completed_sub_goals,
            "remaining_sub_goals": self.remaining_sub_goals,
            "goal_progress": self.goal_progress,
            "constraints_violated": self.constraints_violated,
            "constraint_satisfaction_rate": self.constraint_satisfaction_rate,
            "user_decision": self.user_decision,
            "termination_reason": self.termination_reason,
            "satisfaction_level": self.satisfaction_level,
            "user_reasoning": self.user_reasoning,
            "follow_up_intent": self.follow_up_intent,
        }


@dataclass
class GoalTrajectory:
    """Complete goal-oriented conversation trajectory."""
    conversation_id: str
    seed_query: str
    user_persona: str
    uuid: str

    # Goal tracking
    user_goal: str
    sub_goals: List[str]
    goal_completion_rate: float
    goal_achieved: bool

    # Constraint tracking
    constraints: List[Dict]
    overall_constraint_satisfaction_rate: float

    turns: List[GoalTurn]

    # Final outcome
    total_turns: int
    final_decision: str
    final_satisfaction: float

    # Metadata
    timestamp: str
    agent_model: str
    user_model: str
    dynamically_loaded_servers: List[str]
    timeout_occurred: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "conversation_id": self.conversation_id,
                "uuid": self.uuid,
                "seed_query": self.seed_query,
                "user_persona": self.user_persona,
                "user_goal": self.user_goal,
                "sub_goals": self.sub_goals,
                "goal_completion_rate": self.goal_completion_rate,
                "goal_achieved": self.goal_achieved,
                "constraints": self.constraints,
                "overall_constraint_satisfaction_rate": self.overall_constraint_satisfaction_rate,
                "timestamp": self.timestamp,
                "agent_model": self.agent_model,
                "user_model": self.user_model,
                "timeout_occurred": self.timeout_occurred,
            },
            "turns": [turn.to_dict() for turn in self.turns],
            "summary": {
                "total_turns": self.total_turns,
                "final_decision": self.final_decision,
                "final_satisfaction": self.final_satisfaction,
                "dynamically_loaded_servers": self.dynamically_loaded_servers,
            },
        }

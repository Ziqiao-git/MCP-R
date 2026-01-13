"""Subgoal tracking for goal-oriented evaluation."""

import json
from typing import Dict, List, Optional

from toolgym.llm.base import BaseLLM


def _load_first_json_obj(text: str):
    """Extract first complete JSON object from text."""
    s = text.strip()

    # Remove code block wrappers
    if s.startswith("```"):
        if s.startswith("```json"):
            s = s.split("```json", 1)[1]
        else:
            s = s.split("```", 1)[1]
        s = s.split("```", 1)[0].strip()

    # Find first complete {...}
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        block = s[start:i+1]
                        return json.loads(block)

    return json.loads(s)


class SubgoalTracker:
    """Tracks sub-goal completion throughout conversation."""

    def __init__(self, llm: BaseLLM, query: str, constraints: List[Dict] = None):
        self.llm = llm
        self.query = query
        self.constraints = constraints or []
        self.sub_goals: List[str] = []
        self.completed: List[str] = []
        self.remaining: List[str] = []
        self.violated_constraints: List[str] = []

    async def decompose_query(self) -> List[str]:
        """Use LLM to break user query into 3-6 measurable sub-goals."""
        prompt = f"""You are analyzing a user's query to break it into measurable sub-goals.

USER QUERY:
{self.query}

Your task:
Break this query into clear, measurable sub-goals that represent what information or tasks the user needs.

Each sub-goal should:
- Be specific and measurable (clear when completed)
- Represent one type of information or task needed
- Be achievable through tool usage

EXAMPLES:

Query: "What events are happening in Bodrum next week?"
Sub-goals:
1. Find upcoming events in Bodrum
2. Get event dates and details
3. Identify event locations and venues

Query: "I'm a junior investment analyst... planning a due-diligence trip to San Francisco..."
Sub-goals:
1. Find and book round-trip flights from Austin to San Francisco within budget
2. Assess flight weather risk using TAFs for departure/arrival dates
3. Verify startup's Bitcoin UTXO sat ranges and technical claims
4. Check founders' academic credentials and publication records
5. Analyze public partner company financials and valuation
6. Set up WhatsApp group coordination for the trip team

Now break down the user's query into sub-goals.

Output format (strict JSON):
{{
  "sub_goals": [
    "Sub-goal 1",
    "Sub-goal 2",
    "Sub-goal 3",
    ...
  ]
}}"""

        try:
            response = await self.llm.generate_text(prompt)
            data = _load_first_json_obj(response)
            self.sub_goals = data.get("sub_goals", [])
            self.remaining = self.sub_goals.copy()
            return self.sub_goals

        except Exception as e:
            print(f"⚠️  Error decomposing query: {e}")
            self.sub_goals = [f"Answer the query: {self.query}"]
            self.remaining = self.sub_goals.copy()
            return self.sub_goals

    async def evaluate_progress(self, agent_response: str, tool_calls: List[Dict]) -> Dict:
        """Evaluate which sub-goals were addressed in current turn."""
        if not self.remaining:
            return {
                "completed_this_turn": [],
                "remaining": [],
                "progress": 1.0,
                "reasoning": "All sub-goals completed"
            }

        prompt = f"""You are evaluating progress toward completing a user's query.

ORIGINAL QUERY: {self.query}

REMAINING SUB-GOALS:
{chr(10).join(f"{i+1}. {sg}" for i, sg in enumerate(self.remaining)) if self.remaining else "(All sub-goals completed)"}

AGENT'S RESPONSE THIS TURN:
{agent_response}

TOOLS USED:
{chr(10).join(f"- {tc.get('server', 'unknown')}/{tc.get('tool', 'unknown')}" for tc in tool_calls) if tool_calls else "No tools used"}

Your task:
Determine which (if any) of the REMAINING sub-goals were COMPLETED by this agent response.

A sub-goal is COMPLETED if:
- Agent provided specific, actionable information addressing it
- Information came from tool usage (not agent's internal knowledge)
- User could reasonably act on this information

Output format (strict JSON):
{{
  "completed_this_turn": ["exact sub-goal text from REMAINING SUB-GOALS list", "..."],
  "reasoning": "Brief explanation of what was completed this turn"
}}"""

        try:
            response = await self.llm.generate_text(prompt)
            data = _load_first_json_obj(response)
            completed_this_turn = data.get("completed_this_turn", [])
            for sg in completed_this_turn:
                if sg in self.remaining:
                    self.remaining.remove(sg)
                    self.completed.append(sg)

            return {
                "completed_this_turn": completed_this_turn,
                "remaining": self.remaining.copy(),
                "progress": self.progress_percentage,
                "reasoning": data.get("reasoning", "")
            }

        except Exception as e:
            print(f"⚠️  Error evaluating progress: {e}")
            return {
                "completed_this_turn": [],
                "remaining": self.remaining.copy(),
                "progress": self.progress_percentage,
                "reasoning": ""
            }

    async def evaluate_constraints(self, all_turns_data: List[Dict]) -> Dict:
        """Evaluate constraint violations based on ALL conversation turns."""
        if not self.constraints:
            return {
                "constraints_violated": [],
                "constraint_satisfaction_rate": 1.0,
                "reasoning": "No constraints to check"
            }

        # Aggregate data from all turns
        all_tool_calls = []
        all_servers_used = set()
        all_responses = []
        final_response = ""

        for turn in all_turns_data:
            tool_calls = turn.get("tool_calls", [])
            all_tool_calls.extend(tool_calls)
            for tc in tool_calls:
                server = tc.get("server", "")
                if server:
                    all_servers_used.add(server)

            response = turn.get("agent_response", "")
            if response:
                all_responses.append({
                    "turn": turn.get("turn_number", "?"),
                    "response": response
                })
                final_response = response

        # Format constraints
        constraint_lines = []
        for i, c in enumerate(self.constraints):
            desc = c.get("description", "")
            ctype = c.get("type", "")
            verification = c.get("verification", {})
            verification_str = ", ".join(f"{k}={v}" for k, v in verification.items())
            constraint_lines.append(f"{i+1}. Type: {ctype}\n   Description: {desc}\n   Verification: {verification_str}")

        # Format tool summary
        tool_summary = []
        server_tool_counts = {}
        for tc in all_tool_calls:
            server = tc.get("server", "unknown")
            tool = tc.get("tool", "unknown")
            key = f"{server}/{tool}"
            server_tool_counts[key] = server_tool_counts.get(key, 0) + 1

        for key, count in server_tool_counts.items():
            tool_summary.append(f"- {key} (called {count} time{'s' if count > 1 else ''})")

        prompt = f"""You are evaluating whether the agent violated any constraints.

ORIGINAL QUERY: {self.query}

CONSTRAINTS TO CHECK:
{chr(10).join(constraint_lines)}

CONVERSATION SUMMARY:
- Total turns: {len(all_turns_data)}
- Total tool calls: {len(all_tool_calls)}
- Unique servers: {len(all_servers_used)} ({', '.join(sorted(all_servers_used)) if all_servers_used else 'none'})

TOOL USAGE:
{chr(10).join(tool_summary) if tool_summary else "No tools used"}

LATEST RESPONSE:
{final_response[:1000]}...

Evaluate which constraints were VIOLATED.

Output format (strict JSON):
{{
  "constraints_violated_indices": [1, 3],
  "reasoning": "Explanation of violations"
}}"""

        try:
            response = await self.llm.generate_text(prompt)
            data = _load_first_json_obj(response)
            violated_indices = data.get("constraints_violated_indices", [])
            violated_constraints = []

            for idx in violated_indices:
                constraint_idx = idx - 1
                if 0 <= constraint_idx < len(self.constraints):
                    constraint_obj = self.constraints[constraint_idx]
                    constraint_text = constraint_obj.get("description", "")
                    if constraint_text:
                        violated_constraints.append(constraint_text)

            self.violated_constraints = violated_constraints

            total_constraints = len(self.constraints)
            if total_constraints > 0:
                constraint_rate = 1.0 - (len(self.violated_constraints) / total_constraints)
                constraint_rate = max(0.0, min(1.0, constraint_rate))
            else:
                constraint_rate = 1.0

            return {
                "constraints_violated": violated_constraints,
                "constraint_satisfaction_rate": constraint_rate,
                "reasoning": data.get("reasoning", "")
            }

        except Exception as e:
            print(f"⚠️  Error evaluating constraints: {e}")
            return {
                "constraints_violated": [],
                "constraint_satisfaction_rate": 1.0,
                "reasoning": f"Error: {e}"
            }

    @property
    def progress_percentage(self) -> float:
        """Overall completion percentage (0.0-1.0)."""
        if not self.sub_goals:
            return 0.0
        return len(self.completed) / len(self.sub_goals)

    @property
    def is_complete(self) -> bool:
        """Check if all sub-goals completed."""
        return len(self.remaining) == 0

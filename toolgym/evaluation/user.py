"""Simulated user for goal-oriented evaluation."""

import asyncio
from typing import Dict, List, Optional

from toolgym.llm.base import BaseLLM
from toolgym.evaluation.tracker import SubgoalTracker, _load_first_json_obj


USER_PERSONAS = {
    "curious_researcher": {
        "description": "Academic researcher exploring a topic deeply",
        "behavior": "Asks thoughtful follow-ups, wants comprehensive details",
        "satisfaction_threshold": 0.85,
        "frustration_threshold": 0.3,
        "max_turns": 10,
    },
    "impatient_user": {
        "description": "Busy professional wanting quick answers",
        "behavior": "Gets frustrated quickly if agent is slow or verbose",
        "satisfaction_threshold": 0.8,
        "frustration_threshold": 0.4,
        "max_turns": 5,
    },
    "thorough_analyst": {
        "description": "Detail-oriented analyst verifying accuracy",
        "behavior": "Asks probing questions, checks for inconsistencies",
        "satisfaction_threshold": 0.9,
        "frustration_threshold": 0.25,
        "max_turns": 12,
    },
    "casual_user": {
        "description": "General user with surface-level interest",
        "behavior": "Satisfied with basic answers, doesn't dig deep",
        "satisfaction_threshold": 0.75,
        "frustration_threshold": 0.35,
        "max_turns": 6,
    },
    "skeptical_user": {
        "description": "User who questions agent's capabilities",
        "behavior": "Challenges responses, asks for evidence",
        "satisfaction_threshold": 0.85,
        "frustration_threshold": 0.3,
        "max_turns": 8,
    },
}


class GoalOrientedUser:
    """Simulated user tracking subgoal progress across turns."""

    def __init__(self, llm: BaseLLM, persona_name: str, query: str, subgoal_tracker: SubgoalTracker):
        self.llm = llm
        self.persona_name = persona_name
        self.persona = USER_PERSONAS[persona_name]
        self.query = query
        self.subgoal_tracker = subgoal_tracker

    def _format_tool_usage(self, tool_calls: List[Dict] = None) -> str:
        """Format tool usage information."""
        if not tool_calls:
            return "❌ NO TOOLS USED - Agent answered from internal knowledge"

        tools_list = []
        for tc in tool_calls:
            server = tc.get('server', 'unknown')
            tool = tc.get('tool', 'unknown')
            tools_list.append(f"  ✓ {server}/{tool}")

        return f"Tools used:\n" + "\n".join(tools_list)

    def _format_goal_progress(self, completed: List[str], remaining: List[str], progress_pct: float) -> str:
        """Format goal progress information."""
        lines = [f"Goal Progress: {progress_pct:.0%}", "", "Completed:"]
        if completed:
            for sg in completed:
                lines.append(f"  ✅ {sg}")
        else:
            lines.append("  (none yet)")

        lines.append("")
        lines.append("Remaining:")
        if remaining:
            for sg in remaining:
                lines.append(f"  ⏳ {sg}")
        else:
            lines.append("  (all done!)")

        return "\n".join(lines)

    async def evaluate_and_decide(
        self,
        query: str,
        agent_response: str,
        conversation_history: List[Dict],
        current_turn: int,
        tool_calls: List[Dict],
        completed_sub_goals: List[str],
        remaining_sub_goals: List[str],
        goal_progress: float,
        constraints_violated: List[str],
        constraint_satisfaction_rate: float,
        progress_reasoning: str = ""
    ) -> Dict:
        """Evaluate response and decide next action."""

        history_text = "\n\n".join([
            f"Turn {h['turn']}: {h['query']}\nAgent: {str(h['response'])}"
            for h in conversation_history[-3:]
        ]) if conversation_history else "No previous turns"

        constraint_info = ""
        if constraints_violated:
            constraint_info = f"""
**CONSTRAINT VIOLATIONS:**
❌ {len(constraints_violated)} violated:
{chr(10).join(f"  {i+1}. {cv[:80]}..." for i, cv in enumerate(constraints_violated))}
Satisfaction Rate: {constraint_satisfaction_rate:.0%}
"""
        else:
            constraint_info = f"""
**CONSTRAINT STATUS:**
✅ No violations
Satisfaction Rate: {constraint_satisfaction_rate:.0%}
"""

        prompt = f"""You are a simulated user evaluating an AI agent's response.

**YOUR PERSONA:**
- Name: {self.persona_name}
- Description: {self.persona['description']}
- Behavior: {self.persona['behavior']}

**YOUR ORIGINAL QUERY:**
{self.query}

**SUBGOAL PROGRESS:**
{self._format_goal_progress(completed_sub_goals, remaining_sub_goals, goal_progress)}

{f"Progress this turn: {progress_reasoning}" if progress_reasoning else ""}

{constraint_info}

**CONVERSATION SO FAR:**
{history_text}

**CURRENT TURN ({current_turn}):**
Your Question: {query}

Agent's Response:
{agent_response[:2000]}...

**TOOL USAGE:**
{self._format_tool_usage(tool_calls)}

---

**EVALUATION CRITERIA:**

1. **Subgoal Progress** (Most Important):
   - Did this response help complete any remaining sub-goals?

2. **Tool Usage**:
   - No tools used → subtract 0.3-0.5 from satisfaction

3. **Response Quality**:
   - Is information specific and actionable?

**SATISFACTION CALCULATION:**
- Base satisfaction: Quality of response (0.0-1.0)
- Tool usage penalty: No tools → -0.3 to -0.5
- Subgoal progress bonus: +0.1 to +0.2
- All goals done → +0.2
- Constraint violation penalty: -0.1 to -0.2 each

**TERMINATION DECISION:**
- All sub-goals completed (100%) AND constraints satisfied (≥80%) → TERMINATE "goal_achieved"
- Satisfaction >= {self.persona['satisfaction_threshold']} → TERMINATE "satisfied"
- Satisfaction <= {self.persona['frustration_threshold']} → TERMINATE "frustrated"
- Turn >= {self.persona['max_turns']} → TERMINATE "max_turns"
- Otherwise → CONTINUE

**OUTPUT FORMAT (strict JSON):**
{{
  "decision": "CONTINUE" or "TERMINATE",
  "termination_reason": "goal_achieved|satisfied|frustrated|max_turns|null",
  "follow_up_query": "Your next question..." (if CONTINUE),
  "intent": "goal_progress|clarification|expansion|verification",
  "satisfaction_level": 0.0-1.0,
  "reasoning": "Explain your evaluation (2-3 sentences)"
}}"""

        max_retries = 3
        retry_delay = 2.0
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self.llm.generate_text(prompt)

                if not response or not response.strip():
                    last_error = "Empty response"
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        break

                decision = _load_first_json_obj(response)
                if "decision" not in decision:
                    decision["decision"] = "TERMINATE"
                    decision["termination_reason"] = "error"

                return decision

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    break

        # Default to CONTINUE on failure
        return {
            "decision": "CONTINUE",
            "termination_reason": None,
            "follow_up_query": f"Can you provide more details about {self.subgoal_tracker.remaining[0] if self.subgoal_tracker.remaining else 'my query'}?",
            "intent": "clarification",
            "satisfaction_level": 0.6,
            "reasoning": f"Evaluation failed ({last_error}), continuing"
        }

    async def generate_bonus_question(self, conversation_history: List[Dict]) -> Dict:
        """Generate a bonus question after all goals are achieved."""
        history_text = "\n\n".join([
            f"Turn {h['turn']}: {h['query']}\nAgent: {str(h['response'])}..."
            for h in conversation_history[-3:]
        ]) if conversation_history else "No previous turns"

        prompt = f"""You are a simulated user who just had all their goals completed.

**YOUR ORIGINAL QUERY:**
{self.query}

**CONVERSATION HISTORY:**
{history_text}

**SITUATION:**
All your original goals have been achieved! Generate ONE bonus question to explore a related aspect.

**OUTPUT FORMAT (strict JSON):**
{{
  "bonus_question": "Your bonus question here",
  "constraints": ["constraint 1", "constraint 2", ...],
  "reasoning": "Brief explanation"
}}"""

        try:
            response = await self.llm.generate_text(prompt)
            data = _load_first_json_obj(response)
            return {
                "bonus_question": data.get("bonus_question", ""),
                "constraints": data.get("constraints", [])
            }

        except Exception as e:
            print(f"⚠️  Error generating bonus question: {e}")
            return {"bonus_question": "", "constraints": []}

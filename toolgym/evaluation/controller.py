"""Goal-oriented conversation controller."""

import json
from datetime import datetime
from typing import Dict, List, Optional

from toolgym.evaluation.types import GoalTurn, GoalTrajectory
from toolgym.evaluation.tracker import SubgoalTracker
from toolgym.evaluation.user import GoalOrientedUser
from toolgym.agents.dynamic import DynamicReActAgent


class GoalOrientedController:
    """Orchestrates goal-oriented multi-turn conversations."""

    def __init__(
        self,
        agent: DynamicReActAgent,
        goal_oriented_user: GoalOrientedUser,
        subgoal_tracker: SubgoalTracker,
        max_turns: int,
        query_uuid: str = "unknown",
        enable_bonus_questions: bool = False
    ):
        self.agent = agent
        self.user = goal_oriented_user
        self.subgoal_tracker = subgoal_tracker
        self.max_turns = max_turns
        self.turns: List[GoalTurn] = []
        self.bonus_question_generated = False
        self.query_uuid = query_uuid
        self.enable_bonus_questions = enable_bonus_questions
        self._timeout_occurred = False

    def _format_history(self) -> List[Dict]:
        """Format conversation history for user evaluation."""
        return [
            {
                "turn": turn.turn_number,
                "query": turn.query,
                "response": turn.agent_response,
                "satisfaction": turn.satisfaction_level
            }
            for turn in self.turns
        ]

    def _get_model_name(self, obj) -> str:
        """Safely extract model name."""
        for attr in ['_llm', 'llm']:
            llm = getattr(obj, attr, None)
            if llm is not None:
                config = getattr(llm, 'config', None)
                if config is not None:
                    model_name = getattr(config, 'model', None)
                    if model_name:
                        return model_name
        config = getattr(obj, 'config', None)
        if config is not None:
            model_name = getattr(config, 'model', None)
            if model_name:
                return model_name
        return 'unknown'

    async def run_conversation(self, seed_query: str) -> GoalTrajectory:
        """Run a complete goal-oriented conversation."""
        self._timeout_occurred = False

        print("\n" + "="*70)
        print("🎯 STARTING GOAL-ORIENTED CONVERSATION")
        print("="*70)
        print(f"Seed Query: {seed_query}")
        print(f"Persona: {self.user.persona_name}")
        print("="*70 + "\n")

        self.turns = []

        # Decompose query into sub-goals
        print("📋 Decomposing query into sub-goals...")
        await self.subgoal_tracker.decompose_query()
        print(f"✓ Identified {len(self.subgoal_tracker.sub_goals)} sub-goals:")
        for i, sg in enumerate(self.subgoal_tracker.sub_goals, 1):
            print(f"  {i}. {sg}")
        print()

        current_query = seed_query
        consecutive_no_tool_turns = 0

        for turn_num in range(1, self.max_turns + 1):
            print(f"\n{'='*70}")
            print(f"TURN {turn_num}/{self.max_turns}")
            print(f"{'='*70}\n")

            # Get available tools info
            available_servers = list(getattr(self.agent, '_loaded_servers', set()))
            available_tool_count = len(self.agent.mcp.get_tools()) if hasattr(self.agent, 'mcp') else 0

            print(f"🔧 Available servers: {', '.join(available_servers) if available_servers else 'none'}")
            print(f"🔧 Total tools: {available_tool_count}\n")
            print(f"👤 User Query: {current_query}\n")

            # Agent responds
            print("🤖 Agent thinking...")
            agent_response = await self.agent.run(current_query)
            print(f"\n🤖 Agent Response:\n{agent_response[:500]}...\n")

            # Check for timeout
            if "timed out" in str(agent_response).lower():
                print("🛑 LLM TIMEOUT DETECTED - Stopping")
                self._timeout_occurred = True
                break

            # Extract tool calls from agent state
            turn_tool_calls = []
            if hasattr(self.agent, 'state') and self.agent.state.messages:
                for msg in self.agent.state.messages:
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            turn_tool_calls.append({
                                "server": getattr(tc, 'server', 'unknown'),
                                "tool": tc.name,
                                "arguments": tc.arguments,
                            })

            # Track consecutive no-tool turns
            if not turn_tool_calls:
                consecutive_no_tool_turns += 1
                print(f"⚠️  No tools used ({consecutive_no_tool_turns}/3 consecutive)")
                if consecutive_no_tool_turns >= 3:
                    print("🛑 EARLY STOP: No tools for 3 turns")
                    turn = GoalTurn(
                        turn_number=turn_num,
                        query=current_query,
                        agent_response=agent_response,
                        tool_calls=[],
                        reasoning_trace=[],
                        available_servers=available_servers,
                        available_tool_count=available_tool_count,
                        completed_sub_goals=[],
                        remaining_sub_goals=self.subgoal_tracker.remaining.copy(),
                        goal_progress=self.subgoal_tracker.progress_percentage,
                        constraints_violated=[],
                        constraint_satisfaction_rate=1.0,
                        user_decision="TERMINATE",
                        termination_reason="early_stop_no_tools",
                        satisfaction_level=0.0,
                        user_reasoning="Agent refused to use tools",
                        follow_up_intent=None
                    )
                    self.turns.append(turn)
                    break
            else:
                consecutive_no_tool_turns = 0

            # Evaluate subgoal progress
            print("📊 Evaluating subgoal progress...")
            progress_info = await self.subgoal_tracker.evaluate_progress(
                agent_response, turn_tool_calls
            )

            completed_this_turn = progress_info["completed_this_turn"]
            if completed_this_turn:
                print(f"✅ Completed: {', '.join(completed_this_turn)}")
            else:
                print("⏳ No sub-goals completed")
            print(f"📈 Progress: {progress_info['progress']:.0%}\n")

            # Evaluate constraints
            print("🔒 Evaluating constraints...")
            all_turns_data = []
            for prev_turn in self.turns:
                all_turns_data.append({
                    "turn_number": prev_turn.turn_number,
                    "query": prev_turn.query,
                    "agent_response": prev_turn.agent_response,
                    "tool_calls": prev_turn.tool_calls
                })
            all_turns_data.append({
                "turn_number": turn_num,
                "query": current_query,
                "agent_response": agent_response,
                "tool_calls": turn_tool_calls
            })

            constraint_info = await self.subgoal_tracker.evaluate_constraints(all_turns_data)
            constraints_violated = constraint_info["constraints_violated"]
            constraint_satisfaction_rate = constraint_info["constraint_satisfaction_rate"]

            if constraints_violated:
                print(f"❌ Violations: {len(constraints_violated)}")
            else:
                print("✅ No violations")
            print(f"📊 Constraint rate: {constraint_satisfaction_rate:.0%}\n")

            # User evaluates
            print("👤 User evaluating...")
            decision = await self.user.evaluate_and_decide(
                query=current_query,
                agent_response=agent_response,
                conversation_history=self._format_history(),
                current_turn=turn_num,
                tool_calls=turn_tool_calls,
                completed_sub_goals=self.subgoal_tracker.completed,
                remaining_sub_goals=self.subgoal_tracker.remaining,
                goal_progress=progress_info["progress"],
                constraints_violated=constraints_violated,
                constraint_satisfaction_rate=constraint_satisfaction_rate,
                progress_reasoning=progress_info.get("reasoning", "")
            )

            print(f"👤 Satisfaction: {decision['satisfaction_level']:.2f}")
            print(f"👤 Decision: {decision['decision']}")
            print(f"👤 Reasoning: {decision['reasoning']}\n")

            # Create turn record
            turn = GoalTurn(
                turn_number=turn_num,
                query=current_query,
                agent_response=agent_response,
                tool_calls=turn_tool_calls,
                reasoning_trace=[],
                available_servers=available_servers,
                available_tool_count=available_tool_count,
                completed_sub_goals=completed_this_turn,
                remaining_sub_goals=self.subgoal_tracker.remaining.copy(),
                goal_progress=progress_info["progress"],
                constraints_violated=constraints_violated,
                constraint_satisfaction_rate=constraint_satisfaction_rate,
                user_decision=decision["decision"],
                termination_reason=decision.get("termination_reason"),
                satisfaction_level=decision["satisfaction_level"],
                user_reasoning=decision["reasoning"],
                follow_up_intent=decision.get("intent")
            )
            self.turns.append(turn)

            # Check for bonus question
            if (self.enable_bonus_questions and
                progress_info["progress"] >= 1.0 and
                not self.bonus_question_generated and
                decision["decision"] == "TERMINATE"):

                print("\n🎁 ALL GOALS ACHIEVED! Generating bonus question...")
                bonus_data = await self.user.generate_bonus_question(self._format_history())
                bonus_question = bonus_data.get("bonus_question", "")

                if bonus_question:
                    self.bonus_question_generated = True
                    current_query = bonus_question
                    self.subgoal_tracker.query = bonus_question
                    self.subgoal_tracker.sub_goals = []
                    self.subgoal_tracker.completed = []
                    self.subgoal_tracker.remaining = []
                    await self.subgoal_tracker.decompose_query()
                    self.user.query = bonus_question
                    continue
                else:
                    break

            # Check termination
            if decision["decision"] == "TERMINATE":
                print(f"🛑 Terminated: {decision.get('termination_reason', 'unknown')}")
                break

            # Update for next turn
            current_query = decision.get("follow_up_query", "")
            if not current_query:
                print("⚠️  No follow-up, terminating")
                break

        # Build trajectory
        print("\n" + "="*70)
        print("📝 CONVERSATION SUMMARY")
        print("="*70)
        print(f"Total turns: {len(self.turns)}")
        print(f"Goal completion: {self.subgoal_tracker.progress_percentage:.0%}")

        final_constraint_rate = self.turns[-1].constraint_satisfaction_rate if self.turns else 1.0
        print(f"Final satisfaction: {self.turns[-1].satisfaction_level:.2f}" if self.turns else "Final satisfaction: 0.00")
        print(f"Constraint satisfaction: {final_constraint_rate:.0%}")

        dynamically_loaded = list(getattr(self.agent, '_loaded_servers', set()))

        trajectory = GoalTrajectory(
            conversation_id=f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            seed_query=seed_query,
            user_persona=self.user.persona_name,
            uuid=self.query_uuid,
            user_goal=self.user.query,
            sub_goals=self.subgoal_tracker.sub_goals,
            goal_completion_rate=self.subgoal_tracker.progress_percentage,
            goal_achieved=self.subgoal_tracker.is_complete,
            constraints=self.subgoal_tracker.constraints,
            overall_constraint_satisfaction_rate=final_constraint_rate,
            turns=self.turns,
            total_turns=len(self.turns),
            final_decision=self.turns[-1].user_decision if self.turns else "NONE",
            final_satisfaction=self.turns[-1].satisfaction_level if self.turns else 0.0,
            timestamp=datetime.now().isoformat(),
            agent_model=self._get_model_name(self.agent),
            user_model=self._get_model_name(self.user.llm),
            dynamically_loaded_servers=dynamically_loaded,
            timeout_occurred=self._timeout_occurred
        )

        print("="*70 + "\n")
        return trajectory

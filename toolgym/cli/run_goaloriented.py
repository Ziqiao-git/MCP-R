#!/usr/bin/env python3
"""
Run Goal-Oriented Multi-Turn Agent

Usage:
    # Single query
    python -m toolgym.cli.run_goaloriented "What events are in Bodrum?" \
        --persona curious_researcher

    # From JSON file
    python -m toolgym.cli.run_goaloriented \
        --seeds queries.json \
        --query-index 0 \
        --model gpt-4o-mini \
        --user-model gpt-4o-mini \
        --save-trajectory
"""

import asyncio
import argparse
import json
import os
import uuid as uuid_module
from pathlib import Path
from datetime import datetime

# Load environment from Orchestrator/.env
def _load_env():
    env_path = Path(__file__).parent.parent.parent / "Orchestrator" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

from toolgym import MCPManager, create_llm, DynamicReActAgent, ToolSearchIndex
from toolgym.evaluation import (
    SubgoalTracker,
    GoalOrientedUser,
    GoalOrientedController,
    USER_PERSONAS,
)
from toolgym.evaluation.instructions import AGENT_INSTRUCTION
from toolgym.agents.dynamic import DynamicReActConfig


async def run_single_query(
    query: str,
    query_uuid: str,
    constraints: list,
    args,
    output_dir: Path = None,
):
    """Run a single goal-oriented conversation."""
    print(f"\n{'#'*70}")
    print(f"Query UUID: {query_uuid}")
    print(f"{'#'*70}\n")

    if constraints:
        print(f"📋 Query has {len(constraints)} constraints:")
        for i, c in enumerate(constraints, 1):
            desc = c.get("description", "")
            ctype = c.get("type", "")
            print(f"  {i}. [{ctype}] {desc[:80]}...")

    # Create MCP manager
    mcp = MCPManager()

    # Add meta-mcp server for tool search
    from toolgym.core.types import ServerConfig
    meta_mcp_path = Path(__file__).parent.parent.parent / "tool_retrieval_index" / "server.py"
    if meta_mcp_path.exists():
        mcp.add_config(ServerConfig(
            name="meta-mcp",
            command="python",
            args=[str(meta_mcp_path)],
            transport="stdio",
        ))

    # Load Smithery servers
    remote_servers_path = Path(__file__).parent.parent.parent / "MCP_INFO_MGR" / "mcp_data" / "working" / "remote_servers.json"
    if remote_servers_path.exists():
        mcp.load_configs(remote_servers_path)

    async with mcp:
        # Connect to meta-mcp
        if "meta-mcp" in mcp.available_servers:
            await mcp.connect("meta-mcp")
            print(f"✓ Connected to meta-mcp ({len(mcp.get_tools())} tools)")

        # Create LLMs
        agent_llm = create_llm(args.model, temperature=0.0)
        user_llm = create_llm(args.user_model, temperature=0.0)

        # Create search index using meta-mcp
        search_index = ToolSearchIndex(
            server_path=str(meta_mcp_path),
            mcp_manager=mcp,
        )
        if "meta-mcp" not in mcp.connected_servers:
            await search_index.connect()

        # Create agent
        config = DynamicReActConfig(
            max_iterations=args.max_iterations,
            system_prompt=AGENT_INSTRUCTION,
        )
        agent = DynamicReActAgent(
            llm=agent_llm,
            mcp_manager=mcp,
            search_index=search_index,
            config=config,
        )

        # Create subgoal tracker
        subgoal_tracker = SubgoalTracker(
            llm=user_llm,
            query=query,
            constraints=constraints,
        )

        # Create goal-oriented user
        max_turns = args.max_turns or USER_PERSONAS[args.persona]["max_turns"]
        goal_user = GoalOrientedUser(
            llm=user_llm,
            persona_name=args.persona,
            query=query,
            subgoal_tracker=subgoal_tracker,
        )

        # Create controller
        controller = GoalOrientedController(
            agent=agent,
            goal_oriented_user=goal_user,
            subgoal_tracker=subgoal_tracker,
            max_turns=max_turns,
            query_uuid=query_uuid,
            enable_bonus_questions=args.enable_bonus_questions,
        )

        # Run conversation
        trajectory = await controller.run_conversation(query)

        # Print summary
        print(f"\n{'='*70}")
        print(f"CONVERSATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Turns: {trajectory.total_turns}")
        print(f"Goal Completion: {trajectory.goal_completion_rate:.0%}")
        print(f"Final Satisfaction: {trajectory.final_satisfaction:.2f}")
        print(f"{'='*70}\n")

        # Save trajectory
        if args.save_trajectory and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"trajectory_{query_uuid}_{timestamp}.json"
            with open(output_file, "w") as f:
                json.dump(trajectory.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"💾 Saved trajectory: {output_file}")

        return trajectory


async def main():
    parser = argparse.ArgumentParser(description="Run Goal-Oriented Multi-Turn Agent")
    parser.add_argument("query", nargs="?", help="Initial seed query")
    parser.add_argument("--seeds", help="JSON file with seed queries")
    parser.add_argument("--query-index", type=int, help="Index of query (0-based)")
    parser.add_argument(
        "--persona",
        default="curious_researcher",
        choices=list(USER_PERSONAS.keys()),
        help="Simulated user persona",
    )
    parser.add_argument("--max-turns", type=int, help="Maximum conversation turns")
    parser.add_argument("--model", default="gpt-4o-mini", help="Agent model")
    parser.add_argument("--user-model", default="gpt-4o-mini", help="User model")
    parser.add_argument("--max-iterations", type=int, default=30, help="Max iterations per turn")
    parser.add_argument("--save-trajectory", action="store_true", help="Save trajectory")
    parser.add_argument("--enable-bonus-questions", action="store_true", help="Enable bonus questions")
    parser.add_argument("--pass-number", type=int, default=1, help="Pass number for output organization")

    args = parser.parse_args()

    # Load queries
    if args.seeds:
        with open(args.seeds) as f:
            data = json.load(f)
            items = data.get("items", data) if isinstance(data, dict) else data

        if args.query_index is not None:
            if args.query_index < 0 or args.query_index >= len(items):
                parser.error(f"Query index {args.query_index} out of range")
            items = [items[args.query_index]]
    elif args.query:
        items = [{"query": args.query, "uuid": str(uuid_module.uuid4())}]
    else:
        parser.error("Must provide either 'query' or --seeds file")

    print(f"\n🎯 Goal-Oriented Multi-Turn Agent")
    print(f"{'='*70}")
    print(f"Agent model: {args.model}")
    print(f"User model: {args.user_model}")
    print(f"Persona: {args.persona}")
    print(f"Queries to run: {len(items)}")
    print(f"{'='*70}\n")

    # Setup output directory
    output_dir = None
    if args.save_trajectory:
        model_safe = args.model.split("/")[-1].replace(":", "-")
        pass_folder = f"pass@{args.pass_number}"
        output_dir = Path("trajectories") / "goaloriented" / model_safe / pass_folder

    # Run queries
    for item in items:
        query = item.get("query", "")
        query_uuid = item.get("uuid", str(uuid_module.uuid4()))
        constraints = item.get("constraints", [])

        try:
            await run_single_query(
                query=query,
                query_uuid=query_uuid,
                constraints=constraints,
                args=args,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("✅ All conversations complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())

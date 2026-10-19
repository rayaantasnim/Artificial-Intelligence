import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Callable, Coroutine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MultiAgentOrchestrator")

@dataclass
class AgentState:
    """Explicit state context dictionary/object passed between agents."""
    task_id: str
    objective: str
    plan: List[str] = field(default_factory=list)
    current_step_index: int = 0
    worker_outputs: List[str] = field(default_factory=list)
    status: Literal["PLANNING", "WORKING", "INTERRUPTED", "COMPLETED", "FAILED"] = "PLANNING"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Abstract base class for specialized agents."""
    def __init__(self, name: str):
        self.name = name

    async def execute(self, state: AgentState) -> AgentState:
        raise NotImplementedError


class PlannerAgent(BaseAgent):
    """Planner Agent responsible for decomposing objectives into concrete steps."""
    async def execute(self, state: AgentState) -> AgentState:
        logger.info(f"[{self.name}] Decomposing objective: '{state.objective}'")
        await asyncio.sleep(0.5)  # Simulate LLM reasoning latency
        
        # Simulate generating a structured plan
        state.plan = [
            "Step 1: Perform exploratory data analysis and schema validation.",
            "Step 2: Generate preliminary high-risk transformation rules (Requires Review).",
            "Step 3: Execute data migration and write audit logs."
        ]
        state.current_step_index = 0
        state.status = "WORKING"
        logger.info(f"[{self.name}] Plan generated successfully with {len(state.plan)} steps.")
        return state


class WorkerAgent(BaseAgent):
    """Worker Agent responsible for executing the granular steps in the plan."""
    async def execute(self, state: AgentState) -> AgentState:
        if state.current_step_index >= len(state.plan):
            state.status = "COMPLETED"
            return state

        current_step = state.plan[state.current_step_index]
        logger.info(f"[{self.name}] Executing step [{state.current_step_index}]: {current_step}")
        await asyncio.sleep(0.7)  # Simulate tool execution / computation
        
        output = f"Successfully executed: {current_step}"
        state.worker_outputs.append(output)
        
        # Increment progress
        state.current_step_index += 1
        return state


class Orchestrator:
    """State management machine supporting cyclic execution loops and HITL interrupts."""
    def __init__(self, planner: PlannerAgent, worker: WorkerAgent):
        self.planner = planner
        self.worker = worker
        self.interrupt_callback: Optional[Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]] = None

    def register_interrupt_handler(self, callback: Callable[[AgentState], Coroutine[Any, Any, Dict[str, Any]]]):
        """Registers an async callback to handle human-in-the-loop interventions."""
        self.interrupt_callback = callback

    async def run(self, initial_state: AgentState) -> AgentState:
        logger.info("Orchestration pipeline started.")
        
        while initial_state.status not in ["COMPLETED", "FAILED"]:
            if initial_state.status == "PLANNING":
                initial_state = await self.planner.execute(initial_state)
            
            elif initial_state.status == "WORKING":
                # Check for structural break condition (e.g., reaching a high-risk decision node)
                # For demonstration, we trigger an interrupt at Step index 1 (Step 2 in the plan)
                if initial_state.current_step_index == 1:
                    logger.warning("Structural break triggered: High-risk decision node reached. Pausing for Human-in-the-Loop review.")
                    initial_state.status = "INTERRUPTED"
                    
                    if self.interrupt_callback:
                        # Invoke interactive module to capture steering input
                        steering_patch = await self.interrupt_callback(initial_state)
                        
                        # Apply human overrides safely to the state context
                        if steering_patch:
                            logger.info(f"Applying human steering overrides: {steering_patch}")
                            if "plan" in steering_patch:
                                initial_state.plan = steering_patch["plan"]
                            if "objective" in steering_patch:
                                initial_state.objective = steering_patch["objective"]
                            if "current_step_index" in steering_patch:
                                initial_state.current_step_index = steering_patch["current_step_index"]
                            if "status" in steering_patch:
                                initial_state.status = steering_patch["status"]
                            else:
                                initial_state.status = "WORKING"  # Resume execution by default
                    else:
                        logger.error("No interrupt handler registered. Aborting execution.")
                        initial_state.status = "FAILED"
                        break
                else:
                    initial_state = await self.worker.execute(initial_state)
                    
            elif initial_state.status == "INTERRUPTED":
                # Fallback safeguard if loop resumes without state transition
                initial_state.status = "WORKING"

        logger.info(f"Orchestration pipeline finished with status: {initial_state.status}")
        return initial_state


async def mock_human_interaction_simulation(state: AgentState) -> Dict[str, Any]:
    """Simulates an interactive API abstraction layer capturing granular human steering input."""
    logger.info("--- HITL INTERACTION MODULE ACTIVE ---")
    logger.info(f"Current Task ID: {state.task_id}")
    logger.info(f"Objective: {state.objective}")
    logger.info(f"Current Plan: {state.plan}")
    logger.info(f"Execution Index: {state.current_step_index}")
    
    # Simulate asynchronous wait for human input/API payload
    await asyncio.sleep(1.0)
    
    logger.info("Simulating human operator updating Step 2 and approving resumption...")
    
    # Return steering modifications
    return {
        "plan": [
            state.plan[0],
            "Step 2 (Modified by Human): Execute strict data sanitation and run dry-run validation.",
            state.plan[2]
        ],
        "status": "WORKING"
    }


async def main():
    # Initialize explicit state context
    initial_state = AgentState(
        task_id="TASK-9921-ALPHA",
        objective="Migrate legacy database schema securely with zero downtime."
    )

    # Instantiate specialized agents
    planner = PlannerAgent(name="Planner-Alpha")
    worker = WorkerAgent(name="Worker-Beta")

    # Instantiate orchestrator
    orchestrator = Orchestrator(planner=planner, worker=worker)
    
    # Register the interactive HITL interrupt handler
    orchestrator.register_interrupt_handler(mock_human_interaction_simulation)

    # Execute non-blocking orchestration pipeline via asyncio
    final_state = await orchestrator.run(initial_state)

    print("\n--- FINAL EXECUTION STATE SUMMARY ---")
    print(f"Task ID: {final_state.task_id}")
    print(f"Final Status: {final_state.status}")
    print(f"Final Plan: {final_state.plan}")
    print("Worker Outputs:")
    for idx, output in enumerate(final_state.worker_outputs, start=1):
        print(f"  [{idx}] {output}")


if __name__ == "__main__":
    asyncio.run(main())
"""
Workflow Orchestration Agent — Chain multiple agents together in configurable workflows.

Uses: LangGraph-style state management

Usage:
    from agents.workflow_agent import WorkflowAgent, WorkflowStep
    wf = WorkflowAgent()
    wf.add_step("scrape", scrape_func, inputs=["url"])
    wf.add_step("clean", clean_func, depends_on=["scrape"])
    wf.add_step("summarize", summarize_func, depends_on=["clean"])
    result = wf.run(url="https://example.com")
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class WorkflowStep:
    """A single step in a workflow pipeline."""
    name: str
    func: Callable
    depends_on: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2
    timeout_seconds: int = 60
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: str = None
    duration: float = 0.0


class WorkflowAgent:
    """
    Orchestrate multi-step workflows with dependency resolution.

    Features:
    - Dependency-based execution order
    - Automatic retry on failure
    - Shared state between steps
    - Execution logging and reporting
    """

    def __init__(self, name: str = "Workflow"):
        self.name = name
        self.steps: dict[str, WorkflowStep] = {}
        self.state: dict[str, Any] = {}
        self.execution_log: list[dict] = []

    def add_step(
        self,
        name: str,
        func: Callable,
        depends_on: list[str] = None,
        inputs: list[str] = None,
        max_retries: int = 2,
    ) -> "WorkflowAgent":
        """Add a step to the workflow. Returns self for chaining."""
        self.steps[name] = WorkflowStep(
            name=name,
            func=func,
            depends_on=depends_on or [],
            inputs=inputs or [],
            max_retries=max_retries,
        )
        return self

    def _resolve_execution_order(self) -> list[str]:
        """Topological sort to determine execution order."""
        visited = set()
        order = []
        visiting = set()

        def visit(name: str):
            if name in visiting:
                raise ValueError(f"Circular dependency detected involving: {name}")
            if name in visited:
                return
            visiting.add(name)
            step = self.steps[name]
            for dep in step.depends_on:
                if dep not in self.steps:
                    raise ValueError(f"Step '{name}' depends on unknown step '{dep}'")
                visit(dep)
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in self.steps:
            visit(name)

        return order

    def _execute_step(self, step: WorkflowStep) -> Any:
        """Execute a single workflow step with retry logic."""
        step.status = "running"
        start_time = time.time()

        # Prepare inputs from state and dependencies
        kwargs = {}
        for input_key in step.inputs:
            if input_key in self.state:
                kwargs[input_key] = self.state[input_key]

        for dep in step.depends_on:
            dep_step = self.steps[dep]
            if dep_step.result is not None:
                kwargs[dep] = dep_step.result

        while step.retry_count <= step.max_retries:
            try:
                result = step.func(**kwargs) if kwargs else step.func()
                step.result = result
                step.status = "completed"
                step.duration = time.time() - start_time
                return result
            except Exception as e:
                step.retry_count += 1
                if step.retry_count > step.max_retries:
                    step.status = "failed"
                    step.error = str(e)
                    step.duration = time.time() - start_time
                    raise RuntimeError(f"Step '{step.name}' failed after {step.max_retries} retries: {e}")
                print(f"  ⚠️ Retry {step.retry_count}/{step.max_retries} for '{step.name}': {e}")
                time.sleep(1)

    def run(self, **initial_state) -> dict:
        """
        Execute the full workflow.

        Args:
            **initial_state: Initial state variables (passed as inputs to steps)

        Returns:
            dict with all step results and execution report
        """
        self.state.update(initial_state)

        # Resolve execution order
        order = self._resolve_execution_order()
        
        print(f"\nWorkflow: {self.name}")
        print(f"   Steps: {' -> '.join(order)}")
        print(f"{'-' * 50}")

        start_time = time.time()
        failed_steps = []

        for step_name in order:
            step = self.steps[step_name]
            print(f"\n  ▶ [{step_name}]", end=" ")

            try:
                result = self._execute_step(step)
                print(f"SUCCESS ({step.duration:.1f}s)")

                # Store result in shared state
                self.state[step_name] = result

                self.execution_log.append({
                    "step": step_name,
                    "status": "completed",
                    "duration": step.duration,
                })
            except RuntimeError as e:
                print(f"❌ {e}")
                failed_steps.append(step_name)
                self.execution_log.append({
                    "step": step_name,
                    "status": "failed",
                    "error": str(e),
                    "duration": step.duration,
                })

        total_time = time.time() - start_time

        report = {
            "workflow": self.name,
            "total_steps": len(order),
            "completed": len(order) - len(failed_steps),
            "failed": len(failed_steps),
            "total_time": round(total_time, 2),
            "state": {k: v for k, v in self.state.items() if k in [s.name for s in self.steps.values()]},
            "execution_log": self.execution_log,
        }

        print(f"\n{'-' * 50}")
        status = "Complete" if not failed_steps else f"{len(failed_steps)} failed"
        print(f"  {status} -- {report['completed']}/{report['total_steps']} steps in {total_time:.1f}s")

        return report

    def visualize(self) -> str:
        """Generate a text-based workflow diagram."""
        order = self._resolve_execution_order()
        lines = [f"Workflow: {self.name}", ""]

        for i, name in enumerate(order):
            step = self.steps[name]
            deps = f" (<- {', '.join(step.depends_on)})" if step.depends_on else ""
            status_icon = {"pending": "[ ]", "running": "[/]", "completed": "[x]", "failed": "[!]"}.get(step.status, "[ ]")
            lines.append(f"  {status_icon} Step {i+1}: {name}{deps}")
            if i < len(order) - 1:
                lines.append("       |")

        return "\n".join(lines)


if __name__ == "__main__":
    # Demo: Simple data processing workflow

    def fetch_data(url: str = "https://example.com") -> dict:
        """Step 1: Fetch data."""
        return {"text": "Hello World! This is sample data for processing.", "source": url}

    def clean_data(fetch_data: dict = None) -> dict:
        """Step 2: Clean the data."""
        text = fetch_data["text"] if fetch_data else "no data"
        return {"cleaned_text": text.lower().strip(), "word_count": len(text.split())}

    def analyze_data(clean_data: dict = None) -> dict:
        """Step 3: Analyze the data."""
        text = clean_data["cleaned_text"] if clean_data else ""
        return {
            "char_count": len(text),
            "word_count": clean_data.get("word_count", 0),
            "avg_word_length": round(len(text.replace(" ", "")) / max(clean_data.get("word_count", 1), 1), 1),
        }

    def generate_report(analyze_data: dict = None) -> str:
        """Step 4: Generate final report."""
        if not analyze_data:
            return "No data to report."
        return (
            f"📊 Report: {analyze_data['word_count']} words, "
            f"{analyze_data['char_count']} chars, "
            f"avg word length: {analyze_data['avg_word_length']}"
        )

    # Build and run workflow
    wf = WorkflowAgent(name="Data Processing Pipeline")
    wf.add_step("fetch_data", fetch_data, inputs=["url"])
    wf.add_step("clean_data", clean_data, depends_on=["fetch_data"])
    wf.add_step("analyze_data", analyze_data, depends_on=["clean_data"])
    wf.add_step("generate_report", generate_report, depends_on=["analyze_data"])

    print(wf.visualize())
    result = wf.run(url="https://example.com")

    print(f"\n📄 Final output: {result['state'].get('generate_report', 'N/A')}")

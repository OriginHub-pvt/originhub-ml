"""
AgentBase
=========

Base class for all agents in the system.
"""

from typing import Optional, List
from src.agentic.core.state import State
from src.agentic.ml.inference_engine import InferenceEngine


class AgentBase:
    """
    Base class for all agents.

    Provides:
    - `run()` lifecycle: build prompt → call model → update state
    - error-safe execution
    - tracking last_prompt and last_output
    """

    def __init__(
        self,
        name: str,
        inference_engine: InferenceEngine,
        prompt_builder: object,
        heavy: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ):
        self.name = name
        self.engine = inference_engine
        self.prompt_builder = prompt_builder
        self.heavy = heavy
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

        self.last_prompt: Optional[str] = None
        self.last_output: Optional[str] = None

    def build_prompt(self, state: State) -> str:
        raise NotImplementedError("Subclasses must implement build_prompt().")

    def run(self, state: State) -> State:
        """
        Run the agent: build prompt, call inference engine, store outputs.
        """
        if not hasattr(state, "agent_outputs") or state.agent_outputs is None:
            state.agent_outputs = {}

        try:
            prompt = self.build_prompt(state)
            self.last_prompt = prompt

            # Debug: print agent prompt and a compact state snapshot (conditional)
            try:
                state_snapshot = repr({k: v for k, v in state.__dict__.items() if k in ('input_text','interpreted','clarifications')})
            except Exception:
                state_snapshot = repr(state.__dict__)
            from src.agentic.utils.logger import debug
            debug(f"[AgentBase] >>> Agent: {self.name} — state before run: {state_snapshot}")
            debug(f"[AgentBase] >>> Prompt for {self.name}:\n{prompt}")

            output = self.engine.generate(
                prompt=prompt,
                heavy=self.heavy,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=self.stop,
            )
            self.last_output = output
            state.agent_outputs[self.name] = output
            # Debug: print raw model output (truncated)
            try:
                out_preview = output if isinstance(output, str) else repr(output)
            except Exception:
                out_preview = repr(output)
            if len(out_preview) > 1000:
                out_preview = out_preview[:1000] + '...[truncated]'
            from src.agentic.utils.logger import debug
            debug(f"[AgentBase] <<< Raw output for {self.name}:\n{out_preview}")

            try:
                state_snapshot_after = repr({k: v for k, v in state.__dict__.items() if k in ('input_text','interpreted','clarifications')})
            except Exception:
                state_snapshot_after = repr(state.__dict__)
            debug(f"[AgentBase] >>> Agent: {self.name} — state after run: {state_snapshot_after}")

        except Exception as e:
            error_text = f"error: {type(e).__name__}: {e}"
            self.last_output = error_text
            state.agent_outputs[self.name] = error_text

        return state

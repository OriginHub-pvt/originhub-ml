"""
SummarizerAgent
===============

This agent produces the final user-facing summary by combining:
- interpreted idea
- reviewer output
- mini-review
- strategist output (if new idea)

It uses the light model for fast summarization.

Outputs:
- state.summary
- state.agent_outputs["SummarizerAgent"]
"""

import json
from typing import Any, Dict

from src.agentic.core.agent_base import AgentBase
from src.agentic.core.state import State


class SummarizerAgent(AgentBase):
    """
    SummarizerAgent generates a polished, concise final summary that is sent to the UI.
    """

    def __init__(
        self,
        inference_engine,
        prompt_builder,
        name: str = "SummarizerAgent",
    ):
        """
        Parameters
        ----------
        inference_engine : InferenceEngine
            LLM inference engine (light model).
        prompt_builder : object
            Provides summarizer_prompt(interpreted, analysis, mini_review, strategy).
        name : str
            Key under state.agent_outputs.
        """
        super().__init__(
            name=name,
            inference_engine=inference_engine,
            prompt_builder=prompt_builder,
            heavy=False,           # Summarizer MUST use light model
            max_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            stop=None,
        )

    def build_prompt(self, state: State) -> str:
        """
        Build summarization prompt by passing all relevant state components.

        Parameters
        ----------
        state : State

        Returns
        -------
        str
            Prompt for LLM.
        """
        return self.prompt_builder.summarizer_prompt(
            state.interpreted,
            state.analysis,
            state.mini_review,
            state.strategy,
        )

    def run(self, state: State) -> State:
        """
        Execute summarization:
        - Build prompt
        - Call LLM via AgentBase
        - Parse JSON or fallback to plain text
        - Store final summary into state.summary
        - Store raw output too
        """
        state = super().run(state)
        raw_output = state.agent_outputs.get(self.name, "")

        # If AgentBase recorded an LLM failure
        if raw_output.lower().startswith("error"):
            state.summary = None
            return state

        # Format the plain text output for better readability
        formatted_output = self._format_for_readability(raw_output.strip())
        state.summary = formatted_output
        return state

    def _format_for_readability(self, text: str) -> str:
        """
        Format plain text summary for better readability by:
        - Adding line breaks between major sections
        - Preserving list formatting
        - Adding spacing around section titles
        """
        lines = text.split("\n")
        formatted_lines = []
        
        section_keywords = [
            "EXECUTIVE SUMMARY",
            "OPPORTUNITY ASSESSMENT",
            "COMPETITIVE POSITION",
            "GO-TO-MARKET STRATEGY",
            "CRITICAL SUCCESS FACTORS",
            "IMMEDIATE NEXT STEPS",
            "RISK ASSESSMENT",
            "OVERALL VERDICT",
        ]
        
        for i, line in enumerate(lines):
            line = line.rstrip()
            
            # Check if this line is a section header
            is_section = any(keyword in line.upper() for keyword in section_keywords)
            
            # Add extra spacing before section headers (except the first one)
            if is_section and i > 0 and formatted_lines and formatted_lines[-1].strip():
                formatted_lines.append("")  # Blank line before section
            
            formatted_lines.append(line)
            
            # Add extra spacing after section headers
            if is_section and i + 1 < len(lines):
                # Don't add blank line if next line is already blank
                pass
        
        # Join with newlines and clean up multiple consecutive blank lines
        result = "\n".join(formatted_lines)
        
        # Replace multiple blank lines with single blank line
        while "\n\n\n" in result:
            result = result.replace("\n\n\n", "\n\n")
        return result
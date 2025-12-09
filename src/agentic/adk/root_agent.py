from adk import agent
from src.agentic.core.state import State
from .graph import build_originhub_graph
from types import SimpleNamespace


@agent
class OriginHubRootAgent:

    def __init__(self, inference_engine, prompt_builder, retriever):
        self.graph = build_originhub_graph()
        # Use a SimpleNamespace so ADK wrapped agents can access
        # context members with attribute access (state.context.inference_engine)
        self.context = SimpleNamespace(
            inference_engine=inference_engine,
            prompt_builder=prompt_builder,
            retriever=retriever,
        )

    def run(self, user_text: str):
        # Initialize State object
        state = State(user_input=user_text, context=self.context)

        # Run ADK graph
        final_state = self.graph.run(state)

        return final_state.to_dict()

from adk import Graph
from .wrapped_agents import (
    InterpreterNode,
    ClarifierNode,
    RAGNode,
    EvaluatorNode,
    ReviewerNode,
    StrategistNode,
    SummarizerNode
)


def build_originhub_graph():
    """
    Builds the full conditional pipeline graph using Google ADK.
    Mirrors your python agent pipeline 1:1.
    """

    g = Graph(name="OriginHub-AgentGraph")

    # Register nodes
    g.add_node("interpreter", InterpreterNode())
    g.add_node("clarifier", ClarifierNode())
    g.add_node("rag", RAGNode())
    g.add_node("evaluator", EvaluatorNode())
    g.add_node("reviewer", ReviewerNode())
    g.add_node("strategist", StrategistNode())
    g.add_node("summarizer", SummarizerNode())

    # ---------------------------
    # EDGES (conditional routing)
    # ---------------------------

    # 1. User → Interpreter
    # (this is handled by RootAgent)
    
    # 2. Interpreter always → RAG
    g.add_edge("interpreter", "rag")

    # 3. After RAG → Evaluator
    g.add_edge("rag", "evaluator")

    # 4. Evaluator conditions:

    # Clarification needed
    g.add_edge("evaluator", "clarifier",
               condition="state.need_more_clarification == True")

    # Clarifier loops → Interpreter
    g.add_edge("clarifier", "interpreter")

    # Existing idea (has competitors)
    g.add_edge("evaluator", "reviewer",
               condition="state.is_new_idea == False")

    # New idea (no competitors)
    g.add_edge("evaluator", "strategist",
               condition="state.is_new_idea == True")

    # Both branches → Summarizer
    g.add_edge("reviewer", "summarizer")
    g.add_edge("strategist", "summarizer")

    # Ending node is Summarizer
    g.set_terminal("summarizer")

    return g

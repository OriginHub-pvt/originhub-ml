"""
RAGAgent
========

This agent queries the vector database (Weaviate) to determine whether
the user’s idea already exists. It does NOT call the LLM.

Outputs:
- state.rag_results : list of retrieved matches
- state.is_new_idea : bool
- state.agent_outputs["RAGAgent"] : raw payload or error
"""

from typing import Any, List, Dict

from src.agentic.core.state import State


class RAGAgent:
    """
    RAGAgent queries the vector DB using fields from interpreted idea JSON.
    """

    def __init__(self, vector_db, name: str = "RAGAgent"):
        """
        Parameters
        ----------
        vector_db : object
            Vector DB client implementing `.search(query_text=...)`.
        name : str
            Agent name in state.agent_outputs.
        """
        self.vector_db = vector_db
        self.name = name

    def run(self, state: State) -> State:
        """
        Runs the RAG process:
        - constructs query based on interpreted idea
        - calls the vector DB
        - updates rag_results and is_new_idea
        - stores raw payload
        - gracefully handles exceptions
        """
        if not hasattr(state, "agent_outputs") or state.agent_outputs is None:
            state.agent_outputs = {}

        try:
            interpreted = state.interpreted or {}

            # Simple heuristic: search by title or entire dict
            title = interpreted.get("title", "")
            query_text = title if isinstance(title, str) else str(interpreted)

            # Perform vector search
            results: List[Dict[str, Any]] = self.vector_db.search(query_text=query_text)

            # Update state
            state.rag_results = results
            state.is_new_idea = len(results) == 0

            # Store raw results for debugging/UI
            state.agent_outputs[self.name] = str(results)

        except Exception as e:
            # On DB failure → do not crash
            state.rag_results = []
            state.is_new_idea = False
            state.agent_outputs[self.name] = f"error: {type(e).__name__}: {e}"

        return state
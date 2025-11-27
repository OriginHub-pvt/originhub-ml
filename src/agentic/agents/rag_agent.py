"""
RAGAgent
========

Retrieves similar ideas from Weaviate to detect:
- Whether the idea already exists,
- Competitive landscape,
- Potential duplicates.

This agent does NOT call the LLM.

Outputs stored in State:
- state.rag_results : list of retrieved matches
- state.is_new_idea : bool
- state.agent_outputs["RAGAgent"] : raw retrieval payload or error
"""

import os
from typing import Any, Dict, List
from src.agentic.core.state import State
from src.agentic.utils.logger import debug


class RAGAgent:
    """
    RAGAgent performs semantic search over Weaviate using the interpreted idea.
    """

    def __init__(self, vector_db, name: str = "RAGAgent"):
        """
        Parameters
        ----------
        vector_db : object
            A retriever object implementing `.search(query_text=...)`.
        name : str
            Name used when storing output into state.agent_outputs.
        """
        self.vector_db = vector_db
        self.name = name

    def run(self, state: State) -> State:
        """
        Runs the RAG retrieval step.

        Steps:
        - Extract 'title' field from interpreted idea.
        - Perform semantic search in Weaviate.
        - Update state with results.
        - Gracefully handle retrieval errors.

        Parameters
        ----------
        state : State
            Shared pipeline state object.

        Returns
        -------
        State
            Updated state with retrieval results.
        """
        # Ensure agent_outputs dict exists
        if not hasattr(state, "agent_outputs") or state.agent_outputs is None:
            state.agent_outputs = {}
        try:
            interpreted = state.interpreted or {}
            title = interpreted.get("title", "")

            # Execute vector search
            results: List[Dict[str, Any]] = self.vector_db.search(
                query_text=title
            )

            # Store results into state
            state.rag_results = results
            # Decide whether the idea is "new" using a distance threshold.
            # If there are no results, the idea is new. Otherwise prefer the
            # top similarity distance; lower == more similar. We treat the
            # idea as new when the best (lowest) distance is greater than
            # the configured threshold.
            # Check for error results first
            if results and any("error" in str(r) for r in results):
                # Treat retrieval errors as new ideas (prefer strategist)
                state.is_new_idea = True
                debug(f"[RAGAgent] Error in results, treating as new_idea=True")
            elif len(results) == 0:
                state.is_new_idea = True
                debug(f"[RAGAgent] No results found, is_new_idea=True")
            else:
                # Extract numeric distances only
                distances = [r.get("distance") for r in results if isinstance(r.get("distance"), (int, float))]
                if not distances:
                    # If no numeric distances available, treat as new (prefer strategist)
                    state.is_new_idea = True
                    debug(f"[RAGAgent] No numeric distances found, treating as new_idea=True")
                else:
                    top = min(distances)
                    try:
                        threshold = float(os.getenv("RAG_NEW_THRESHOLD", 0.35))
                    except Exception:
                        threshold = 0.35
                    state.is_new_idea = top > threshold
                    # Debug info: show top distance and threshold for diagnosis
                    debug(f"[RAGAgent] top_distance={top}, threshold={threshold}, is_new_idea={state.is_new_idea}")
            state.agent_outputs[self.name] = str(results)

        except Exception as e:
            # Fail gracefully without breaking pipeline
            # When RAG fails, treat as new idea (prefer strategist over reviewer with bad data)
            state.rag_results = []
            state.is_new_idea = True
            state.agent_outputs[self.name] = (
                f"error: {type(e).__name__}: {e}"
            )
            debug(f"[RAGAgent] Error while searching: {e}, treating as new_idea=True")

        return state

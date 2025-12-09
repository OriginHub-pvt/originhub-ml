"""
Weaviate RAG Client
===================

Lightweight wrapper around Weaviate's text search API.
Provides a simple .search() interface compatible with RAGAgent.
"""

from typing import List, Dict, Any
import weaviate


class WeaviateRAGClient:
    """
    Minimal RAG client providing a normalized search() API for agents.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str,
        text_property: str = "text",
    ):
        """
        Parameters
        ----------
        endpoint : str
            URL of the Weaviate instance.
        api_key : str
            API key or bearer token for authentication.
        index_name : str
            Weaviate collection/class name used for RAG.
        text_property : str
            Field name containing the text content.
        """
        self.index_name = index_name
        self.text_property = text_property

        # Auth config based on OpenAI-compatible headers
        auth = weaviate.auth.AuthApiKey(api_key)

        self.client = weaviate.Client(
            url=endpoint,
            auth_client_secret=auth
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using Weaviate nearText.

        Parameters
        ----------
        query : str
            Query text used for similarity search.
        top_k : int
            Number of nearest results to return.

        Returns
        -------
        List[Dict[str, Any]]
            List of results with keys:
            - "text": str
            - "score": float (similarity/distance)
        """
        try:
            response = (
                self.client.query
                .get(
                    self.index_name,
                    [self.text_property, "_additional {distance}"]
                )
                .with_near_text({"concepts": [query]})
                .with_limit(top_k)
                .do()
            )

            raw = response.get("data", {}).get("Get", {}).get(self.index_name, [])
            if not raw:
                return []

            normalized = []
            for item in raw:
                text_val = item.get(self.text_property)
                distance = item.get("_additional", {}).get("distance")

                if text_val is None:
                    continue

                normalized.append({
                    "text": text_val,
                    "score": distance
                })

            return normalized

        except Exception:
            # Gracefully degrade on network errors, schema issues, etc.
            return []

"""
WeaviateRetriever
=================

A thin wrapper around a Weaviate client + collection.

This retriever exposes a unified `.search(query_text=...)` interface
that RAGAgent can use without depending on Weaviate internals.

Outputs match the expected format:
[
    {
        "id": "<uuid>",
        "title": "...",
        "summary": "...",
        "distance": 0.12
    }
]
"""

from typing import List, Dict, Any


class WeaviateRetriever:
    """
    WeaviateRetriever performs semantic search using `.query.near_text()`.
    """

    def __init__(self, client, collection_name: str):
        """
        Parameters
        ----------
        client : weaviate.WeaviateClient
            A connected Weaviate Python client.
        collection_name : str
            Target collection containing stored ideas/articles.
        """
        self.client = client
        self.collection_name = collection_name
        self.collection = client.collections.get(collection_name)

    def search(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the collection.

        Parameters
        ----------
        query_text : str
            Query used to find similar stored ideas.

        Returns
        -------
        List[Dict[str, Any]]
            Retrieved objects with title, summary and distance.
        """
        # Return error if client is not connected
        if self.client is None:
            return [{"error": "Weaviate client not initialized. Check connection and HUGGINGFACE_APIKEY."}]
        
        try:
            response = self.collection.query.near_text(
                query=query_text,
                limit=5,
            )

            results = []
            for obj in response.objects:
                results.append(
                    {
                        "id": obj.uuid,
                        "title": obj.properties.get("title", ""),
                        "summary": obj.properties.get("summary", ""),
                        "distance": getattr(obj.metadata, 'distance', getattr(obj, 'distance', None)),  # lower = more similar
                    }
                )

            return results

        except Exception as e:
            # Return structured error for pipeline
            return [{"error": f"{type(e).__name__}: {e}"}]

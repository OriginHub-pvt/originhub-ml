"""
Tests for WeaviateRAGClient.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agentic.ml.rag_client import WeaviateRAGClient


@patch("src.agentic.ml.rag_client.weaviate")
def test_rag_client_initialization(mock_weaviate):
    """Client should initialize Weaviate client with endpoint and API key."""
    
    mock_weaviate.Client.return_value = MagicMock()

    client = WeaviateRAGClient(
        endpoint="https://weaviate.example.com",
        api_key="XYZ",
        index_name="Ideas",
        text_property="text"
    )

    mock_weaviate.Client.assert_called_once()
    assert client.index_name == "Ideas"
    assert client.text_property == "text"


@patch("src.agentic.ml.rag_client.weaviate")
def test_rag_client_search_calls_weaviate(mock_weaviate):
    """search() should call Weaviate's query API correctly."""

    mock_client = MagicMock()
    mock_weaviate.Client.return_value = mock_client

    # Mock a typical Weaviate response structure
    mock_client.query.get.return_value.with_near_text.return_value.with_limit.return_value.do.return_value = {
        "data": {
            "Get": {
                "Ideas": [
                    {"text": "Founders need help", "_additional": {"distance": 0.12}},
                    {"text": "Startup assistant", "_additional": {"distance": 0.20}},
                ]
            }
        }
    }

    client = WeaviateRAGClient(
        endpoint="https://x",
        api_key="Y",
        index_name="Ideas",
        text_property="text"
    )

    results = client.search("founders", top_k=2)

    # Validate result format
    assert len(results) == 2
    assert results[0]["text"] == "Founders need help"
    assert "score" in results[0]

    # Validate low-level Weaviate call chain
    mock_client.query.get.assert_called_once_with(
        "Ideas",
        ["text", "_additional {distance}"]
    )


@patch("src.agentic.ml.rag_client.weaviate")
def test_rag_client_handles_empty_results(mock_weaviate):
    """search() should return empty list when no matches."""

    mock_client = MagicMock()
    mock_weaviate.Client.return_value = mock_client

    # Empty Weaviate response
    mock_client.query.get.return_value.with_near_text.return_value.with_limit.return_value.do.return_value = {
        "data": {"Get": {"Ideas": []}}
    }

    client = WeaviateRAGClient(
        endpoint="https://x",
        api_key="Y",
        index_name="Ideas",
        text_property="text"
    )

    results = client.search("something", top_k=5)
    assert results == []


@patch("src.agentic.ml.rag_client.weaviate")
def test_rag_client_gracefully_handles_weaviate_errors(mock_weaviate):
    """search() should not throw on Weaviate errors; returns empty list."""

    mock_client = MagicMock()
    mock_weaviate.Client.return_value = mock_client

    # Make query.raise_exception
    mock_client.query.get.side_effect = Exception("Connection error")

    client = WeaviateRAGClient(
        endpoint="https://x",
        api_key="Y",
        index_name="Ideas",
        text_property="text"
    )

    results = client.search("query", top_k=3)

    assert results == []

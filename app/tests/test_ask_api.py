"""
Test cases for the /ask endpoint.
"""
import os
import sys
import pytest
from unittest.mock import patch, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .test_base import BaseTestAPI

class TestAskQuestionAPI(BaseTestAPI):
    """Test cases for the /ask endpoint."""
    
    def test_ask_question_success(self):
        """Test asking a question successfully."""
        test_question = "What is the test about?"
        
        with patch("app.api.v1.endpoints.RAGService") as mock_rag_service:
            mock_service = mock_rag_service.return_value
            mock_service.answer_query = AsyncMock(
                return_value=(
                    "This is a test answer.", 
                    [{"source": "Source 1"}, {"source": "Source 2"}]
                )
            )
            
            response = self.client.post(
                "/api/v1/ask",
                json={"question": test_question, "top_k": 3}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a test answer."
        assert len(data["sources"]) == 2
        assert data["sources"][0] == "Source 1"

    def test_ask_question_missing_question(self):
        """Test asking with missing question."""
        response = self.client.post(
            "/api/v1/ask",
            json={"top_k": 3}  # Missing question
        )
        assert response.status_code == 422  # Validation error

    @patch("app.api.v1.endpoints.RAGService")
    def test_ask_question_processing_error(self, mock_rag_service):
        """Test error during question processing."""
        mock_service = mock_rag_service.return_value
        mock_service.answer_query = AsyncMock(side_effect=Exception("Processing failed"))
        
        response = self.client.post(
            "/api/v1/ask",
            json={"question": "What is this?", "top_k": 3}
        )
        
        assert response.status_code == 500
        assert "Error processing your question" in response.json()["detail"]

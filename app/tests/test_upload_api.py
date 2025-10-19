"""
Test cases for the /upload endpoint.
"""
import os
import sys
import pytest
from unittest.mock import patch, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .test_base import BaseTestAPI

class TestUploadDocumentAPI(BaseTestAPI):
    """Test cases for the /upload endpoint."""
    
    def test_upload_document_success(self):
        """Test successful document upload."""
        test_file = self.create_test_file("test.txt")
        
        with open(test_file, "rb") as f:
            response = self.client.post(
                "/api/v1/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.txt"
        assert data["status"] == "ingested"

    def test_upload_document_missing_file(self):
        """Test upload with missing file."""
        response = self.client.post("/api/v1/upload", files={})
        assert response.status_code == 422  # Validation error

    @patch("app.api.v1.endpoints.IngestionService")
    def test_upload_document_processing_error(self, mock_ingestion_service):
        """Test error during document processing."""
        mock_service = mock_ingestion_service.return_value
        mock_service.ingest_file = AsyncMock(side_effect=Exception("Processing failed"))
        
        test_file = self.create_test_file("test.txt")
        
        with open(test_file, "rb") as f:
            response = self.client.post(
                "/api/v1/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == 500
        assert "Error processing document" in response.json()["detail"]

    @pytest.mark.parametrize("filename,content_type", [
        ("test.pdf", "application/pdf"),
        ("test.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("test.txt", "text/plain"),
    ])
    def test_upload_different_file_types(self, filename, content_type):
        """Test uploading different file types."""
        test_file = self.create_test_file(filename)
        
        with open(test_file, "rb") as f:
            response = self.client.post(
                "/api/v1/upload",
                files={"file": (filename, f, content_type)}
            )
        # print(response.data)
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == filename
        assert data["status"] == "ingested"

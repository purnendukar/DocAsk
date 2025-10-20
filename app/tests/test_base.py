"""
Base test class with common setup and utilities for API tests.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.main import app

class BaseTestAPI:
    """Base test class with common setup and utilities."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Common setup for all test methods."""
        self.client = TestClient(app)
        self.tmp_path = tmp_path
        self.test_text = "This is a test document content for unit testing."
        yield
        # Teardown if needed

    def create_test_file(self, filename, content=None):
        """Helper to create a test file."""
        content = content or self.test_text
        file_path = self.tmp_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix.lower() == '.pdf':
            return self._create_test_pdf(file_path, content)
            
        file_path.write_text(content)
        return file_path
        
    def _create_test_pdf(self, file_path, content):
        """Helper to create a simple PDF file for testing."""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        c = canvas.Canvas(str(file_path), pagesize=letter)
        text = c.beginText(40, 750)  # Position text
        text.setFont("Helvetica", 12)
        
        # Split content into lines that fit the page width
        for line in content.split('\n'):
            text.textLine(line)
            
        c.drawText(text)
        c.save()
        return file_path

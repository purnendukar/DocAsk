"""
Pydantic schemas for DocAsk API.
Defines request and response models for document upload and Q&A endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# -------------------------------
# Upload Document Response
# -------------------------------
class UploadResponse(BaseModel):
    """Response after uploading and processing a document."""
    document_id: str = Field(..., description="Unique ID assigned to the uploaded document")
    filename: str = Field(..., description="Original name of the uploaded file")
    status: str = Field(..., description="Processing status (e.g., 'success', 'failed')")
    message: Optional[str] = Field(None, description="Additional info or error message")


# -------------------------------
# Ask (Question) Request
# -------------------------------
class AskRequest(BaseModel):
    """Request body for asking a question."""
    question: str = Field(..., description="User's question")
    top_k: int = Field(3, description="Number of most relevant document chunks to consider (default: 3)")


# -------------------------------
# Ask (Question) Response
# -------------------------------
class AskResponse(BaseModel):
    """Response containing the model's answer and context sources."""
    answer: str = Field(..., description="Generated answer from the RAG pipeline")
    sources: List[str] = Field(..., description="List of document snippets used to answer the question")
    relevant_docs: Optional[List[str]] = Field(None, description="Optional list of relevant document IDs or filenames")

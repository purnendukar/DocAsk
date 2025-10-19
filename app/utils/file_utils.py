import io
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import PyPDF2
from docx import Document

logger = logging.getLogger(__name__)

def extract_text_from_file(file_path: str) -> Tuple[str, Optional[str]]:
    """Extract text from a file based on its extension.
    
    Args:
        file_path: Path to the file to extract text from
        
    Returns:
        Tuple of (extracted_text, error_message)
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        with open(file_path, 'rb') as f:
            if ext == '.pdf':
                return _extract_text_from_pdf(f), None
            elif ext == '.docx':
                return _extract_text_from_docx(f), None
            elif ext in ('.txt', '.md', '.csv', '.json'):
                # Try to read as text file
                try:
                    return f.read().decode('utf-8'), None
                except UnicodeDecodeError:
                    return "", "Could not decode file as UTF-8 text"
            else:
                return "", f"Unsupported file type: {ext}"
                
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}", exc_info=True)
        return "", str(e)

def _extract_text_from_pdf(file_obj) -> str:
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file_obj)
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
        return ""

def _extract_text_from_docx(file_obj) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = Document(io.BytesIO(file_obj.read()))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}", exc_info=True)
        return ""

def save_uploaded_file(upload_file, upload_dir: Path) -> Optional[Path]:
    """Save an uploaded file to disk."""
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / upload_file.filename
        
        with open(file_path, 'wb') as f:
            f.write(upload_file.file.read())
            
        return file_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}", exc_info=True)
        return None

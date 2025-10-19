# DocAsk API Documentation

This document provides detailed information about the DocAsk API endpoints, request/response formats, and usage examples.

## Base URL
All API endpoints are relative to the base URL:
```
http://localhost:8000/api/v1
```

## Authentication
Currently, the API does not require authentication. All endpoints are publicly accessible.

## Endpoints

### 1. Health Check

#### GET /health

Check if the API is running.

**Response**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. Document Upload

#### POST /upload

Upload and process a document for the RAG system. The document will be processed and stored in the vector database.

**Request Headers**
```
Content-Type: multipart/form-data
```

**Request Body**
| Parameter | Type   | Required | Description                          |
|-----------|--------|----------|--------------------------------------|
| file      | file   | Yes      | Document file to process (PDF, DOCX, TXT) |

**Example Request**
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@document.pdf"
```

**Success Response (200 OK)**
```json
{
  "document_id": "unique-document-id-123",
  "filename": "document.pdf",
  "status": "ingested",
  "message": "Document processed successfully"
}
```

**Error Responses**
- 400 Bad Request: Missing filename or invalid file format
- 500 Internal Server Error: Error processing the document

### 3. Ask a Question

#### POST /ask

Ask a question to the RAG system. The system will retrieve relevant information from the ingested documents and generate an answer.

**Request Headers**
```
Content-Type: application/json
```

**Request Body**
```json
{
  "question": "What is the main topic of the document?",
  "top_k": 3
}
```

**Parameters**
| Parameter | Type    | Required | Default | Description                           |
|-----------|---------|----------|---------|---------------------------------------|
| question  | string  | Yes      | -       | The question to ask                   |
| top_k     | integer | No       | 3       | Number of relevant chunks to consider |

**Success Response (200 OK)**
```json
{
  "answer": "The main topic of the document is...",
  "sources": [
    "...relevant text excerpt 1...",
    "...relevant text excerpt 2..."
  ],
  "relevant_docs": null
}
```

**Error Responses**
- 400 Bad Request: Invalid request parameters
- 500 Internal Server Error: Error processing the question

## Data Models

### Upload Response
```typescript
{
  document_id: string;  // Unique ID assigned to the uploaded document
  filename: string;     // Original name of the uploaded file
  status: string;       // Processing status (e.g., 'ingested', 'failed')
  message?: string;     // Additional info or error message
}
```

### Ask Request
```typescript
{
  question: string;  // User's question
  top_k?: number;    // Number of relevant chunks to consider (default: 3)
}
```

### Ask Response
```typescript
{
  answer: string;     // Generated answer from the RAG pipeline
  sources: string[];  // List of document snippets used to answer the question
  relevant_docs?: string[];  // Optional list of relevant document IDs or filenames
}
```

## Error Handling

### Error Response Format
```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Status Codes
- 400: Bad Request - Invalid input or missing required parameters
- 404: Not Found - Requested resource not found
- 500: Internal Server Error - Something went wrong on the server side

## Rate Limiting
Currently, there are no rate limits implemented, but they may be added in the future.

## Versioning
- Current API version: v1
- Version is included in the URL path: `/api/v1/...`

## Changelog

### v1.0.0 (2025-10-19)
- Initial release of the DocAsk API
- Added document upload and question-answering endpoints

# DocAsk

DocAsk is a powerful document question-answering system built with FastAPI, leveraging state-of-the-art language models and vector embeddings for accurate and efficient document retrieval and question-answering capabilities.

## 🚀 Features

- **Document Ingestion**: Upload and process various document formats (PDF, DOCX, etc.)
- **Vector Embeddings**: Utilizes FAISS for efficient similarity search
- **RAG Pipeline**: Implements Retrieval-Augmented Generation for accurate answers
- **RESTful API**: Easy integration with other services
- **Docker Support**: Containerized for easy deployment

## 🛠️ Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Poetry (for local development)

## 🚀 Quick Start

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DocAsk.git
   cd DocAsk
   ```

2. Copy the example environment file and update with your settings:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Build and start the services:
   ```bash
   docker-compose up --build -d
   ```

4. Access the API at `http://localhost:8000`

### Local Development

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the development server:
   ```bash
   poetry run uvicorn app.main:app --reload
   ```

## 📚 API Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`

## 🌐 API Endpoints

- `POST /api/v1/upload`: Upload and process documents
- `POST /api/v1/ask`: Ask question related to doc

## 🧪 Testing

Run tests using pytest:
```bash
poetry run pytest
```

## 🧰 Project Structure

```
DocAsk/
├── app/                    # Application source code
│   ├── api/                # API routes
│   ├── core/               # Core configurations
│   ├── models/             # Database models
│   ├── services/           # Business logic
│   │   ├── ingestion.py    # Document ingestion logic
│   │   ├── llm.py          # Language model interactions
│   │   ├── rag.py          # RAG pipeline
│   │   └── vector_store.py # Vector database operations
│   └── main.py             # FastAPI application
├── tests/                  # Test files
├── .env.example            # Example environment variables
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Docker configuration
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Poetry](https://python-poetry.org/)

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev libomp-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel "poetry==2.1.2"

WORKDIR /app
COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false

# Install deps with fallback for CPU torch
RUN poetry install --no-root --no-interaction --no-ansi -vvv || \
    pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    poetry install --no-root --no-interaction --no-ansi

COPY . .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

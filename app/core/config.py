from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # -------------------------------
    # App settings
    # -------------------------------
    app_name: str = Field("DocAsk", env="APP_NAME")
    app_env: str = Field("development", env="APP_ENV")
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")

    # -------------------------------
    # HuggingFace Embeddings
    # -------------------------------
    hf_model_name: str = Field("all-MiniLM-L6-v2", env="HF_MODEL_NAME")
    hf_batch_size: int = Field(32, env="HF_BATCH_SIZE")

    # -------------------------------
    # File storage / vector store
    # -------------------------------
    vector_store_path: Path = Field(default=Path("./data/vector_store"), env="VECTOR_STORE_PATH")
    upload_dir: Path = Field(default=Path("./data/uploads"), env="UPLOAD_DIR")

    # -------------------------------
    # Database (optional)
    # -------------------------------
    database_url: str = Field("sqlite:///./data/docask.db", env="DATABASE_URL")

    # -------------------------------
    # Logging
    # -------------------------------
    log_level: str = Field("info", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # ✅ Allow extra environment variables


# Global settings instance
settings = Settings()

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Local storage paths
    data_dir: str = "data"
    vectorstore_dir: str = "data/chroma_db"

    # External movie metadata service for the Agent
    imdbot_base_url: str = "https://imdb.iamidiotareyoutoo.com"

    class Config:
        # Load environment variables from .env
        env_file = ".env"


settings = Settings()
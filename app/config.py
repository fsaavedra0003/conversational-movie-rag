from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    data_dir: str = "data"
    vectorstore_dir: str = "data/chroma_db"

    imdbot_base_url: str = "https://imdb.iamidiotareyoutoo.com"

    class Config:
        env_file = ".env"


settings = Settings()
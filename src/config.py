import os


class Settings:
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_endpoint: str = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    data_dir: str = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
    papers_dir: str = os.getenv("PAPERS_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "papers"))
    output_dir: str = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "output"))

    @property
    def llm_available(self) -> bool:
        return bool(self.llm_api_key)


settings = Settings()

from pathlib import Path
from typing import Optional

from openai import OpenAI
from qdrant_client import QdrantClient

from .config import API_KEY, URL_BASE

lib_path = Path(__file__).parent


class RAGalicClient:
    _instance: Optional["RAGalicClient"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, path: str = str(lib_path / "qdrant_db"), **kwargs):
        if self._initialized:
            return
        self._client = QdrantClient(path=path, **kwargs)
        self._client.set_model("jinaai/jina-embeddings-v3")
        self._client.set_sparse_model("Qdrant/bm25")
        self._initialized = True
        print("Successfully connected to Qdrant.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Явное закрытие — предотвращает ошибку при завершении"""
        if hasattr(self, "_client") and self._client:
            self._client.close()
            self._client = None

    @property
    def client(self):
        return self._client


class OpenAIClient:
    _instance: Optional["OpenAIClient"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, api_key: str = API_KEY, base_url: str = URL_BASE):
        if not self._initialized:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self._initialized = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Явное закрытие — предотвращает ошибку при завершении"""
        if hasattr(self, "_client") and self._client:
            self._client.close()
            self._client = None

    @property
    def client(self):
        return self._client


# Глобальный экземпляр (ленивая инициализация)
def get_ragalic_client(**kwargs) -> RAGalicClient:
    return RAGalicClient(**kwargs)


def get_openai_client(**kwargs) -> OpenAIClient:
    return OpenAIClient(**kwargs)

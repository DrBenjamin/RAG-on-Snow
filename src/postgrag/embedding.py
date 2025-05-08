### `src/postgrag/embedding.py`
### LLM class for Snowflake
### Open-Source, hosted on https://github.com/DrBenjamin/RAG-on-Snow
### Please reach out to ben@seriousbenentertainment.org for any questions
"""
Embedding class for PostgreSQL using raglite.
"""
import logging
from raglite.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class PostgresEmbeddings(Embeddings):
    """
    Embedding wrapper for raglite's OpenAIEmbeddings or other supported models.
    """

    def __init__(self, model_name="text-embedding-ada-002", api_key=None):
        self._emb = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)

    def embed_documents(self, texts):
        return self._emb.embed_documents(texts)

    def embed_query(self, text):
        return self._emb.embed_query(text)
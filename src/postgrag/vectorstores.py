### `src/postgrag/vectorstores.py`
### LLM class for Snowflake
### Open-Source, hosted on https://github.com/DrBenjamin/RAG-on-Snow
### Please reach out to ben@seriousbenentertainment.org for any questions
"""
VectorStore implementation for PostgreSQL using raglite.
"""
import logging
from raglite.vectorstores import PGVectorStore
from langchain_core.vectorstores import VectorStore
logger = logging.getLogger(__name__)

class PostgresVectorStore(VectorStore):
    """
    Wrapper around raglite's PGVectorStore for PostgreSQL.
    """

    def __init__(self, connection_string, embedding, table="langchain_pg", vector_length=1024):
        self._store = PGVectorStore(
            conn_str=connection_string,
            table=table,
            embedding=embedding,
            vector_length=vector_length
        )
        self._embedding = embedding

    def add_texts(self, texts, metadatas=None, **kwargs):
        # Adding texts to the vector store
        return self._store.add_texts(texts, metadatas or [{} for _ in texts])

    def similarity_search(self, query, k=8, **kwargs):
        # Performing similarity search
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query, k=8, **kwargs):
        # Performing similarity search with score
        return self._store.similarity_search_with_score(query, k=k)

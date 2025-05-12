### `langchain_snowrag/vectorstores.py`
### VectorStore class for Snowflake
### Open-Source, hosted on https://github.com/DrBenjamin/RAG-on-Snow
### Please reach out to ben@seriousbenentertainment.org for any questions
from __future__ import annotations
import streamlit as st
import hashlib
import json
import logging
import warnings
import os
from typing import Any, Iterable, List, Optional, Tuple, Type
from snowflake.snowpark import Session
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from snowflake.connector import DictCursor
from snowflake.connector.connection import SnowflakeConnection
logger = logging.getLogger(__name__)

# Setting the user agent for Snowflake
os.environ["USER_AGENT"] = "RAG-on-Snow/1.0 (contact: ben@seriousbenentertainment.org)"

# Setting the default vector length
VECTOR_LENGTH = st.session_state.get("option_vector_length", 1024)

# Creating the `SnowflakeVectorStore` class
class SnowflakeVectorStore(VectorStore):
    """Wrapper around Snowflake vector data type used as vector store."""

    def __init__(
        self,
        table: str | list[str],
        connection: Optional[SnowflakeConnection],
        embedding: Embeddings,
        vector_length: int = VECTOR_LENGTH,
    ):

        @st.cache_resource
        def create_session():
            session = Session.builder.configs(st.secrets.snowflake).create()
            try:
                session.use_role(st.secrets.snowflake["role"])
                session.sql(
                    f'USE WAREHOUSE "{st.secrets.snowflake["warehouse"]}"')
                session.use_database(st.secrets.snowflake["database"])
                session.use_schema(st.secrets.snowflake["schema"])
            except Exception as e:
                st.error(f"Error: {e}")
            return session

        if not connection:
            connection = create_session().connection

        if not isinstance(embedding, Embeddings):
            warnings.warn("embeddings input must be Embeddings object.")

        self._connection = connection
        self._table = st.session_state.get("option_table", table)
        self._embedding = embedding
        self._vector_length = vector_length

        # Creating table if in single‐table mode
        if isinstance(self._table, str):
            self.create_table_if_not_exists()

    def create_table_if_not_exists(self) -> None:
        self._connection.cursor().execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table}
            (
              rowid INTEGER AUTOINCREMENT,
              rowhash VARCHAR,
              text VARCHAR,
              metadata VARIANT,
              text_embedding vector(float, {self._vector_length})
            )
            ;
            """
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore index.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        max_id = (
            self._connection.cursor(DictCursor)
            .execute(f"SELECT NVL(max(rowid), 0) as rowid FROM {self._table}")
            .fetchone()["ROWID"]  # type: ignore
        )
        embeds = self._embedding.embed_documents(list(texts))
        if not metadatas:
            metadatas = [{} for _ in texts]

        # Including the database table name in each metadata
        for m in metadatas:
            m["db_table"] = self._table

        data_input = [
            (text, json.dumps(metadata), embed)
            for text, metadata, embed in zip(texts, metadatas, embeds)
        ]
        self._connection.cursor().execute("begin")
        for row in data_input:
            _hash = hashlib.sha256(row[0].encode("UTF-8")).hexdigest()
            _text = row[0].replace("'", "\\'")
            _metadata = row[1]
            _vec = row[2]

            # Building Snowflake array literal from Python list
            vec_literal = "ARRAY_CONSTRUCT(" + ",".join(str(v)
                                                        for v in _vec) + ")"

            _q = f"""
                MERGE INTO {self._table} t USING (
                    SELECT
                        ?::VARCHAR as rowhash,
                        ?::VARCHAR as text,
                        PARSE_JSON(?) as metadata,
                        {vec_literal}::VECTOR(float, {self._vector_length}) as text_embedding
                ) s
                ON s.rowhash = t.rowhash
                WHEN NOT MATCHED THEN
                    INSERT (rowhash, text, metadata, text_embedding)
                    VALUES (s.rowhash, s.text, s.metadata, s.text_embedding);
            """
            self._connection.cursor().execute(_q, [_hash, _text, _metadata])
        self._connection.cursor().execute("commit")

        # Pulling every ids we just inserted
        results = self._connection.cursor(DictCursor).execute(
            f"SELECT rowid FROM {self._table} WHERE rowid > {max_id}"
        )
        return [row["ROWID"] for row in results]  # type: ignore

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, tables: Optional[List[str]] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        # convert embedding list into Snowflake array literal
        embed_literal = "ARRAY_CONSTRUCT(" + ",".join(str(v)
                                                      for v in embedding) + ")"

        # Determining lookup tables: explicit `tables` overrides; else use self._table directly if list
        if tables is not None:
            lookup_tables = tables
        elif isinstance(self._table, list):
            lookup_tables = self._table
        else:
            lookup_tables = [self._table]
        if len(lookup_tables) > 1:
            union_q = "\nUNION ALL\n".join(
                [f"SELECT text, metadata, text_embedding FROM {t}" for t in lookup_tables]
            )
            from_clause = f"(\n{union_q}\n) AS e"
        else:
            from_clause = f"{lookup_tables[0]} e"

        sql_query = f"""
            WITH search_t AS (
                SELECT {embed_literal}::VECTOR(float, {self._vector_length}) AS search_embedding
            )
            SELECT
                text,
                metadata,
                VECTOR_COSINE_SIMILARITY(e.text_embedding, s.search_embedding) AS similarity
            FROM {from_clause}, search_t s
            ORDER BY similarity DESC
            LIMIT {k}
        """
        cursor = self._connection.cursor(DictCursor)
        cursor.execute(sql_query)
        results = cursor.fetchall()

        documents = []
        for row in results:
            metadata = json.loads(row["METADATA"]) or {}  # type: ignore
            doc = Document(page_content=row["TEXT"], metadata=metadata)
            documents.append((doc, row["SIMILARITY"]))  # type: ignore

        return documents

    def similarity_search(
        self, query: str, k: int = 8, tables: Optional[List[str]] = None, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, tables=tables
        )
        return [doc for doc, _ in documents]

    def similarity_search_with_score(
        self, query: str, k: int = 8, tables: Optional[List[str]] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""

        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, tables=tables
        )
        return documents

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 8, tables: Optional[List[str]] = None, **kwargs: Any
    ) -> List[Document]:
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, tables=tables
        )
        return [doc for doc, _ in documents]

    @classmethod
    def from_texts(
        cls: Type[SnowflakeVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        table: Optional[str] = None,
        connection: Optional[SnowflakeConnection] = None,
        **kwargs: Any,
    ) -> SnowflakeVectorStore:
        """Return VectorStore initialized from texts and embeddings.
        Always checks if the table already contains rows; if so, skips writing new data.
        """

        # Using `st.session_state.option_table` for table name
        if table is None:
            table = st.session_state.get("option_table", "LANGCHAIN_TEST")

        # Creating the vector store (table is created if not exists)
        vector_store = cls(table=table, connection=connection,
                           embedding=embedding)  # type: ignore

        # Checking if the table already contains any rows
        cursor = vector_store._connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {vector_store._table}")
        row_count = cursor.fetchone()[0]  # type: ignore
        if row_count == 0:
            vector_store.add_texts(texts=texts, metadatas=metadatas)
        else:
            logger.info(
                f"Table {vector_store._table} already contains data. Skipping new data insertion.")

        return vector_store

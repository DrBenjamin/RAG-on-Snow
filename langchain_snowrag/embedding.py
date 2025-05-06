### `langchain_snowrag/embedding.py`
### Embedding class for Snowflake
### Open-Source, hosted on https://github.com/DrBenjamin/RAG-on-Snow
### Please reach out to ben@seriousbenentertainment.org for any questions
import streamlit as st
import logging
import os
from typing import Any, List, Mapping, Optional
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from snowflake.connector import DictCursor
from snowflake.connector.connection import SnowflakeConnection
logger = logging.getLogger(__name__)

# Setting the user agent for Snowflake
os.environ["USER_AGENT"] = "RAG-on-Snow/1.0 (contact: ben@seriousbenentertainment.org)"

# Creating the `SnowflakeEmbeddings` class
class SnowflakeEmbeddings(BaseModel, Embeddings):
    model_config = {
        "protected_namespaces": (),
        "arbitrary_types_allowed": True
    }
    """Snowflake runs large language models.

    Example:
        .. code-block:: python

            from langchain_snowpoc.embeddings import SnowflakeEmbeddings
            sf_emb = SnowflakeEmbeddings(
                model="e5-base-v2",
            )
            r1 = sf_emb.embed_documents(
                [
                    "Alpha is the first letter of Greek alphabet",
                    "Beta is the second letter of Greek alphabet",
                ]
            )
            r2 = sf_emb.embed_query(
                "What is the second letter of Greek alphabet"
            )

    """

    connection: SnowflakeConnection = None  # type: ignore
    """Connection to Snowflake to use"""

    model: str = st.session_state.get(
        "option_embedding_model", "multilingual-e5-large")
    """Model name to use."""

    show_progress: bool = False
    """Whether to show a tqdm progress bar. Must have `tqdm` installed."""

    model_kwargs: Optional[dict] = None
    """Other model keyword args"""



    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model": self.model, "connection": self.connection},
            **self._default_params,  # type: ignore
        }

    def _process_emb_response(self, input: str) -> List[float]:
        """Process a response from Snowflake.

        Args:
            input: The string to embed.

        Returns:
            The embedding as a list of floats.
        """
        # Using parameterized query to avoid SQL injection and unicode escape issues
        q = "SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_1024(?, ?) as EMBEDDING"
        cursor = self.connection.cursor(DictCursor)
        try:
            cursor.execute(q, (self.model, input))
            result = cursor.fetchone()
            return result["EMBEDDING"]
        finally:
            cursor.close()

    def _embed(self, input: List[str]) -> List[List[float]]:
        if self.show_progress:
            try:
                from tqdm import tqdm

                iter_ = tqdm(input, desc="SnowflakeEmbeddings")
            except ImportError:
                logger.warning(
                    "Unable to show progress bar because tqdm could not be imported. "
                    "Please install with `pip install tqdm`."
                )
                iter_ = input
        else:
            iter_ = input
        return [self._process_emb_response(prompt) for prompt in iter_]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Snowflake's embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = [text for text in texts]
        embeddings = self._embed(instruction_pairs)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using Snowfake embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self._embed([text])[0]
        return embedding

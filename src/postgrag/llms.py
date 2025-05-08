### `src/postgrag/llms.py`
### LLM class for Snowflake
### Open-Source, hosted on https://github.com/DrBenjamin/RAG-on-Snow
### Please reach out to ben@seriousbenentertainment.org for any questions
"""
LLM class for PostgreSQL RAG using raglite.
"""
import logging
from raglite.llms import OpenAIChatLLM  # or another supported LLM
from langchain_core.language_models.llms import LLM

logger = logging.getLogger(__name__)

class PostgresLLM(LLM):
    """
    LLM wrapper for raglite's OpenAIChatLLM or other supported models.
    """

    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        self._llm = OpenAIChatLLM(model=model_name, openai_api_key=api_key)

    @property
    def _llm_type(self):
        return "postgres_llm"

    def _call(self, prompt, stop=None, **kwargs):
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self._llm(prompt)
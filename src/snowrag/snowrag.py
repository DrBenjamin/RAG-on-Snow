### `src/snowrag/snowrag.py`
### Embedding class for Snowflake
### Open-Source, hosted on https://github.com/DrBenjamin/RAG-on-Snow
### Please reach out to ben@seriousbenentertainment.org for any questions
import os
import streamlit as st
import logging
from snowflake.snowpark import Session
logger = logging.getLogger(__name__)


def set_snowflake_user_agent():
    """
    Setting the user agent for Snowflake.
    """
    os.environ["USER_AGENT"] = "RAG-on-Snow/1.0 (contact: ben@seriousbenentertainment.org)"


@st.cache_resource
def create_session():
    """
    Creating the Snowflake session using Streamlit cache.
    """
    session = Session.builder.configs(st.secrets.snowflake).create()
    try:
        session.use_role(st.secrets.snowflake["role"])
        session.sql(f'USE WAREHOUSE "{st.secrets.snowflake["warehouse"]}"')
        session.use_database(st.secrets.snowflake["database"])
        session.use_schema(st.secrets.snowflake["schema"])
    except Exception as e:
        st.error(f"Error: {e}")
        logger.error(f"Error creating Snowflake session: {e}")
    return session


# Function to fetch tables with retry on token expiration
def fetch_tables_with_retry(snowflake_connection):
    """
    Fetching tables from Snowflake with retry on token expiration.
    """
    try:
        return snowflake_connection.cursor().execute(
            "SHOW TABLES").fetchall()  # type: ignore
    except Exception as e:
        err_msg = str(e)
        err_code = getattr(e, "errno", None)
        if (
            ("Authentication token has expired" in err_msg)
            or (err_code == 390114)
            or ("390114" in err_msg)
        ):
            st.warning("Snowflake-Session abgelaufen. Stelle Verbindung wieder her...")
            st.cache_resource.clear()
            snowflake_connection = create_session().connection
            try:
                return snowflake_connection.cursor().execute(
                    "SHOW TABLES").fetchall()  # type: ignore
            except Exception as e2:
                st.error(f"Fehler beim Abrufen der Tabellen nach Re-Login: {e2}")
                logger.error(f"Error fetching tables after re-login: {e2}")
                return []
        else:
            st.error(f"Fehler beim Abrufen der Tabellen: {e}")
            logger.error(f"Error fetching tables: {e}")
            return []


# Function to drop a table
def drop_table_with_retry(snowflake_connection, db_table_name):
    """
    Dropping a table in Snowflake with error handling.
    """
    try:
        snowflake_connection.cursor().execute(
            f"DROP TABLE IF EXISTS {db_table_name}").fetchall()
    except Exception as e:
        logger.error(f"Error dropping table {db_table_name}: {e}")
        raise


# Function to reset the vector store
def _reset_vector_store():
    # Removing any old vector/store/docs so we re‑init against the new table
    for key in ("vector", "docs", "embeddings", "loader"):
        if key in st.session_state:
            del st.session_state[key]
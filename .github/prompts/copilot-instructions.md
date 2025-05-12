# Copilot Instructions for RAG‑on‑Snow

Dearest Copilot,
this project is a Python/Streamlit app that integrates LangChain with Snowflake
Cortex. When generating code snippets or explanations, please follow these guidelines:

1. Output always in Markdown.
2. When referring to a file in this repo, link using `#file:<relative_path>`.
   - Example: [snowrag.py](#file:snowrag.py)  
   - Other common files:
     - [langchain_snowrag/vectorstores.py](#file:langchain_snowrag/vectorstores.py)
     - [langchain_snowrag/embedding.py](#file:langchain_snowrag/embedding.py)
     - [langchain_snowrag/llms.py](#file:langchain_snowrag/llms.py)

3. Code‑block format for changes or new files:
    ````python
    // filepath: #file:<relative_path>
    # ...existing code...
    def my_new_function(...):
        ...
    # ...existing code...
    ````

4. Comments format:
   - Use `#` for comments
   - Start comments with 'Setting', 'Creating', 'Adding', 'Updating' etc.
     (always the gerund form)

5. Adhere to PEP 8:
   - 4‑space indentation, snake_case names
   - Imports at the top of the file
   - Docstrings in Google or NumPy style

6. Preserve existing patterns:
   - Use `@st.cache_resource` for expensive initializations
   - Store and retrieve state via `st.session_state.get("key", default)`
   - Build Snowflake session with `create_session()` in [snowrag.py](#file:snowrag.py)
   - Leverage `SnowflakeEmbeddings`, `Cortex`, `SnowflakeVectorStore`

7. File I/O:
   - Use `os.path.join(...)` and `os.makedirs(..., exist_ok=True)`
   - Handle missing directories before writing files

8. SQL & Snowflake:
   - Use parameterized queries (`?` placeholders)
   - Wrap DDL/DML in code blocks labeled `sql`
   - Example:
     ```sql
     CREATE TABLE IF NOT EXISTS MY_TABLE (
       id INTEGER AUTOINCREMENT,
       data VARCHAR
     );
     ```

9. Duplicate‑image detection in [snowrag.py](#file:snowrag.py):
   - Compute an MD5 hash of `image_bytes`
   - Track seen hashes in a `set` to skip duplicates

10. Error handling & logging:
   - Import and configure `logger = logging.getLogger(__name__)`
   - Raise clear exceptions on invalid inputs

11. Testing:
    - Add or update tests under `tests/`
    - Use `pytest` fixtures to mock `st.session_state` and Snowflake connections
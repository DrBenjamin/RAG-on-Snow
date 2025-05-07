# Change Log

All notable changes to the project will be documented in this file.

## [0.2.0]
Snowflake vs. PostgreSQL + MinIO integration
    - Snowflake DB integration
        - stores unstructured data from documents and other sources as embeddings
    - Snowlfake stage integration
        - stores documents
    - PostgreSQL DB integration
        - stores unstructured data from documents and other sources as embeddings
    - MinIO storage integration
        - stores documents
## [0.1.0]
Initial release
    - RAG on Snow
        - Snowflake DB integration
            - stores unstructured data from documents and other sources as embeddings
            - uses `Langchain` to query this data
    - Streamlit App
        - uses `StreamlitChatMessageHistory` for memory of the Snowflake Cortex generated answers
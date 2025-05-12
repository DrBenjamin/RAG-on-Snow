# RAG on Snow

This application stores unstructured data from documents and other sources like
images and video in a Snowflake DB utilizing Langchain to query this data. The
application is built using Streamlit and uses `StreamlitChatMessageHistory` for
memory of the Snowflake Cortex generated embeddings.

## Langchain implementation (langchain-snowrag)

[Tutorial on Medium](https://medium.com/snowflake/integrating-langchain-with-snowflake-cortex-0367f934f1c1).
See the [documentation](https://docs.snowflake.com/user-guide/snowflake-cortex/vector-embeddings#snowflake-python-connector)
on Snowflake for more information.

## Setup

First create an environment using the `environment.yml` file:

```bash
# Creating a conda environment
conda env create --file environment.yml

# Activate the environment
conda activate snowrag
```

Now configure the Streamlit app in the `.streamlit/secrets.toml` file:

```toml
# Configuring LLM
[LLM]
LLM_CHATBOT_NAME = "<chatbot_name>"
LLM_SYSTEM = "Please write a short answer."
LLM_SYSTEM_PLUS = "Prioritize the most relevant information from the similarity search!"
LLM_ASSISTANT = "How can I help?"
LLM_USER_EXAMPLE = "<user_example>"
LLM_ASSISTANT_EXAMPLE = "<>assistant_example>"

# Configuring Snowflake
[snowflake]
user = "<user_name>"
account = "<account_name>"
private_key_file = "<path to rsa_key.p8>"
role = "<role_name>"
warehouse = "<warehouse_name>"
database = "<database_name>"
schema = "<schema_name>"

# Configuring MinIO storage
[MinIO]
endpoint = "http://127.0.0.1:9000"
bucket = "<bucket_name>"
access_key = "<access_key>"
secret_key = "<secret_key>"
```

## Streamlit web app

Just run the following code:

```bash
# Running the Streamlit app
python -m streamlit run snowrag.py

# Checking the Streamlit app
lsof -i :8501
```

# langchain-snowrag

> **NOTE**: This is just a PoC, not production-ready and covers just a few use cases. [Tutorial on Medium](https://medium.com/snowflake/integrating-langchain-with-snowflake-cortex-0367f934f1c1). See the [documentation](https://docs.snowflake.com/user-guide/snowflake-cortex/vector-embeddings#snowflake-python-connector) on Snowflake for more information.

This is a PoC of how can Cortex be used with Langchain.

It shows how easy it is to integrate Langchain and Cortex.

## Setup

```bash
conda env create --file environment.yml
```

To use the environment you have to activate it. Feel free to check the [documentation](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html).

## Streamlit

Just run the following code:

```bash
python -m streamlit run snowrag.py
```

Wait some time for document to be downloaded and chunked and ask questions regarding it.

For example:

* What is the document about?
* Who is mentioned in the document?
* When was the document created?

import streamlit as st
import logging
import os
import sys
import time
from snowflake.connector import connect
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_snowpoc.embedding import SnowflakeEmbeddings
from langchain_snowpoc.llms import Cortex
from langchain_snowpoc.vectorstores import SnowflakeVectorStore
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

MODEL_LLM = "mistral-large"
MODEL_EMBEDDINGS = "e5-base-v2"
VECTOR_LENGTH = 768

def load_private_key(path):
    with open(path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            None,
            backend=default_backend()
        )
    return private_key

@st.cache_resource
def get_connection():
    p_key = load_private_key("./rsa_key.p8")
    p_key_bytes = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return connect(
        user="bengross_tech",
        account="sv04740.west-europe.azure",
        private_key=p_key_bytes,
        role="HEALTH_DEV",
        warehouse="COMPUTE_WH",
        database="DB_BG_HEALTH",
        schema="PUBLIC"
    )

snowflake_connection = get_connection()

if "vector" not in st.session_state:

    st.session_state.embeddings = SnowflakeEmbeddings(
        connection=snowflake_connection, model=MODEL_EMBEDDINGS
    )

    #"https://paulgraham.com/greatwork.html"
    st.session_state.loader = WebBaseLoader("https://www.gwq-serviceplus.de/ueber-uns")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs
    )
    st.session_state.vector = SnowflakeVectorStore.from_documents(
        st.session_state.documents,
        st.session_state.embeddings,
        vector_length=VECTOR_LENGTH,
    )

st.title("Chat with Docs - Snowflake Edition")

llm = Cortex(connection=snowflake_connection, model=MODEL_LLM)

prompt = ChatPromptTemplate.from_template(
    """
Schreibe einen Abschnitt für eine Anzeige beim Bundesamt für Soziale Sicherung.
über die Verwendung von Sozialdaten in der Cloud. 
Denke schrittweise, bevor Du den Abschnitt schreibst.
<context>
{context}
</context>

Zusätzliche Informationen: {input}"""
)

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Zusätzliche Informationen:")


# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print(f"Antwortszeit: {time.process_time() - start}")

    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

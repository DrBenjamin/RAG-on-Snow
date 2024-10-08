import streamlit as st
import logging
import sys
import time
import os
import fnmatch
import warnings
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module='snowflake.connector'
)
from typing import List
from snowflake.connector import connect
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, Docx2txtLoader, CSVLoader, PyPDFLoader, TextLoader
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
    p_key = load_private_key(st.secrets.snowflake["private_key_file"])
    p_key_bytes = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return connect(
        user=st.secrets.snowflake["user"],
        account=st.secrets.snowflake["account"],
        private_key=p_key_bytes,
        role=st.secrets.snowflake["role"],
        warehouse=st.secrets.snowflake["warehouse"],
        database=st.secrets.snowflake["database"],
        schema=st.secrets.snowflake["schema"]
    )

snowflake_connection = get_connection()

st.title("RAG LLM - Snowflake Edition")

with st.spinner("Processing documents..."):
    if "vector" not in st.session_state:
        class CustomDirectoryLoader:
            def __init__(self, urls, directory_path: str, glob_pattern: str = "*.*"):
                """
                Initialize the loader with a directory path and a glob pattern.
                :param directory_path: Path to the directory containing files to load.
                :param glob_pattern: Glob pattern to match files within the directory.
                :param mode: Mode to use with UnstructuredFileLoader ('single', 'elements', or 'paged').
                """
                self.urls = urls
                self.directory_path = directory_path
                self.glob_pattern = glob_pattern

            def load(self) -> List[Document]:
                """
                Load all files matching the glob pattern in the directory using UnstructuredFileLoader.
                :return: List of Document objects loaded from the files.
                """
                documents = []
                patterns = self.glob_pattern.split('|')
                
                # Construct the full glob pattern
                full_glob_pattern = f"{self.directory_path}{self.glob_pattern}"
                
                # Iterate over all files matched by the glob pattern using os.walk and fnmatch
                for root, dirs, files in os.walk(self.directory_path):
                    st.write("Files: ", files)
                    for filename in files:
                        for pattern in patterns:
                            if fnmatch.fnmatch(filename, pattern):
                                file_path = os.path.join(root, filename)
                                if file_path.endswith(".docx"):
                                    loader = Docx2txtLoader(file_path=file_path)
                                if file_path.endswith(".csv"):
                                    loader = CSVLoader(file_path=file_path)
                                if file_path.endswith(".pdf"):
                                    loader = PyPDFLoader(file_path=file_path)
                                if file_path.endswith(".txt"):
                                    loader = TextLoader(file_path=file_path)
                                docs = loader.load()
                                documents.extend(docs)
                st.write("URLs: ", self.urls)
                for url in self.urls:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    documents.extend(docs)
                return documents
    
        st.session_state.start = time.time()
        st.session_state.embeddings = SnowflakeEmbeddings(
            connection=snowflake_connection, model=MODEL_EMBEDDINGS
        )
        st.session_state.loader = CustomDirectoryLoader(urls=["https://www.gwq-serviceplus.de/ueber-uns", "https://docs.streamlit.io/get-started"], directory_path="../Documents/", glob_pattern="*.docx|*.pdf|*.csv|*.txt")

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
        st.success(f"Documents processed in {int(time.time() - st.session_state.start)} seconds!")

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
    st.session_state.start  = time.time()
    response = retrieval_chain.invoke({"input": prompt})
    st.write(f"{response['answer']} (processed in {int(time.time() - st.session_state.start)} seconds.)")

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

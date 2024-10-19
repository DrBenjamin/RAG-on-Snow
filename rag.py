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
from snowflake.snowpark import Session
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, Docx2txtLoader, CSVLoader, PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_snowrag.embedding import SnowflakeEmbeddings
from langchain_snowrag.llms import Cortex
from langchain_snowrag.vectorstores import SnowflakeVectorStore
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
def create_session():
    session = Session.builder.configs(st.secrets.snowflake).create()
    try:
        session.use_role(st.secrets.snowflake["role"])
        session.sql(f'USE WAREHOUSE "{st.secrets.snowflake["warehouse"]}"')
        session.use_database(st.secrets.snowflake["database"])
        session.use_schema(st.secrets.snowflake["schema"])
    except Exception as e:
        st.error(f"Error: {e}")
    return session

snowflake_connection = create_session().connection

with st.form("document_form"):
    st.title("RAG LLM - Snowflake Edition")
    system_message = st.text_input("System Message")
    folder = os.path.abspath(os.path.join(os.getcwd(), '..'))
    options_offline_resources = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    st.session_state.option_offline_resources = st.selectbox("Offline Resources", options_offline_resources)
    st.session_state.online_resources = st.text_area("Online Resources")
    rag_perform = st.form_submit_button("Submit")
    if rag_perform:
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

                        # Iterate over all files matched by the glob pattern using os.walk and fnmatch
                        st.write("Documents")
                        for root, dirs, files in os.walk(self.directory_path):
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
                                        st.write(file_path)
                                        docs = loader.load()
                                        documents.extend(docs)
                        st.write("Online")
                        if len(self.urls) > 0:
                            for url in self.urls:
                                st.write(url)
                                loader = WebBaseLoader(url)
                                docs = loader.load()
                                documents.extend(docs)
                        return documents
            
                st.session_state.start = time.time()
                st.session_state.embeddings = SnowflakeEmbeddings(
                    connection=snowflake_connection, model=MODEL_EMBEDDINGS
                )
                folder = os.path.abspath(os.path.join(os.getcwd(), '..', st.session_state.option_offline_resources))
                urls = st.session_state.online_resources.split(', ')
                print(urls)
                st.session_state.loader = CustomDirectoryLoader(urls=urls, directory_path=folder, glob_pattern="*.docx|*.pdf|*.csv|*.txt")

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
            {system}
            <context>
            {context}
            </context>

            Task: {input}"""
        )

        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = st.session_state.vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        prompt = st.text_input("Question:")

        # If the user hits enter
        if prompt:
            # Then pass the prompt to the LLM
            st.session_state.start  = time.time()
            input_data = {
                "input": prompt, 
                "system": system_message}
            response = retrieval_chain.invoke(input_data)
            st.write(f"{response['answer']} (processed in {int(time.time() - st.session_state.start)} seconds.)")

            # With a streamlit expander
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")

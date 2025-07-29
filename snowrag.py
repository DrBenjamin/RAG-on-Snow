### `snowrag.py`
### Main application for the Snowflake RAG LLM
### Open-Source, hosted on https://github.com/DrBenjamin/RAG-on-Snow
### Please reach out to ben@seriousbenentertainment.org for any questions
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from langchain_snowrag.vectorstores import SnowflakeVectorStore
from langchain_snowrag.minio import upload_file_to_minio
from langchain_snowrag.llms import Cortex
from langchain_snowrag.embedding import SnowflakeEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader, Docx2txtLoader, CSVLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from snowflake.snowpark import Session
from typing import Union, List, Optional
from minio import Minio
import hashlib
import fitz  # PyMuPDF
import streamlit as st
import logging
logger = logging.getLogger(__name__)
import sys
import time
import os
import io
import requests
import json
import fnmatch
import warnings
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module='snowflake.connector'
)
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Setting the page config
st.set_page_config(
    page_title=f"{st.secrets['LLM']['LLM_CHATBOT_NAME']}",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Detecting iframe embedding using JS and/or 'embed' query parameter (legacy API)
try:
    query_param = st.query_params.get_all("angular")
    is_embed_param = query_param[0].lower() == "true"
    st.session_state["IS_EMBED"] = bool(is_embed_param)
except Exception as e:
    st.session_state["IS_EMBED"] = False

# Setting the user agent for Snowflake
os.environ["USER_AGENT"] = "RAG-on-Snow/1.0 (contact: ben@seriousbenentertainment.org)"

# Setting session states
if "response" not in st.session_state:
    st.session_state.response = None
if "image_extraction" not in st.session_state:
    st.session_state.image_extraction = False
if "offline_resources" not in st.session_state:
    st.session_state.option_offline_resources = os.path.abspath(
        os.path.join(os.getcwd(), '..'))
if "embedding_model" not in st.session_state:
    st.session_state.option_embedding_model = "multilingual-e5-large"
if "vector_length" not in st.session_state:
    st.session_state.option_vector_length = 1024

# Function to load the private key
def load_private_key(path):
    with open(path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            None,
            backend=default_backend()
        )
    return private_key


# Function to create the Snowflake session
@st.cache_resource
def create_session():
    # Logging
    logger.warning(f"Versuche Verbindung zu Snowflake mit Account: {st.secrets.snowflake.get('account', 'UNBEKANNT')}")
    try:
        session = Session.builder.configs(st.secrets.snowflake).create()
        session.use_role(st.secrets.snowflake["role"])
        session.sql(f'USE WAREHOUSE "{st.secrets.snowflake["warehouse"]}"')
        session.use_database(st.secrets.snowflake["database"])
        session.use_schema(st.secrets.snowflake["schema"])
    except Exception as e:
        logger.error(f"Fehler beim Verbindungsaufbau zu Snowflake: {e}")
        st.error(
            f"Fehler beim Verbindungsaufbau zu Snowflake: {e}\n"
            "Bitte prüfe die Account-URL und Zugangsdaten in st.secrets.snowflake.\n"
            "Account-URL sollte z.B. so aussehen: 'xy12345.eu-central-1' (ohne https:// und ohne .snowflakecomputing.com)"
        )
        raise
    return session


# Function to reset the vector store
def _reset_vector_store():
    # Removing any old vector/store/docs so we re‑init against the new table
    for key in ("vector", "docs", "embeddings", "loader"):
        if key in st.session_state:
            del st.session_state[key]


# Function to list all files one level up and open them
def show_open_file_button(filename, source, idx):
    # Using the MinIO URL as the download source
    minio_url = source
    key = f"minio-download-{filename}-{idx}"
    if minio_url:
        if st.session_state["IS_EMBED"] and minio_url.startswith(("http://", "https://")):
            # Creating a direct download link for iOS/iframe users
            st.markdown(
                f'<a href="{minio_url}" download="{filename}" target="_blank" rel="noopener" style="font-weight:bold;">'
                f'📥 Datei {filename} herunterladen</a>',
                unsafe_allow_html=True
            )
            st.info("Falls der Download nicht automatisch startet, bitte den Link lange drücken und 'Download' wählen.")
        else:
            try:
                # Downloading the file into an in-memory buffer (desktop/normal browser)
                response = requests.get(minio_url, verify=False)
                response.raise_for_status()
                buffer = io.BytesIO(response.content)
                st.download_button(
                    label=f"📥 Datei {filename} herunterladen",
                    data=buffer,
                    file_name=filename,
                    mime=None,
                    key=key
                )
            except Exception as e:
                st.error(f"Fehler beim Herunterladen von MinIO: {e}")
        return True
    return False

## Function to ensure the output key chain
def ensure_output_key_chain(result):
    """
    Ensures the chain output has an 'output' key for LangChain callbacks.
    """
    if isinstance(result, dict):
        if "output" not in result:
            # Patch: copy 'answer' to 'output' if present
            if "answer" in result:
                result["output"] = result["answer"]
            else:
                # Fallback: use first string value
                for v in result.values():
                    if isinstance(v, str):
                        result["output"] = v
                        break
    return result


# Creating the Snowflake session
snowflake_connection = create_session().connection

# Adding sidebar for options
with st.sidebar:
    st.title("Optionen")
    st.write("Wähle die Parameter aus.")

    # Adding a selectbox for the user to select the table or create a new one
    def fetch_tables_with_retry():
        global snowflake_connection
        try:
            return snowflake_connection.cursor().execute(
                "SHOW TABLES").fetchall()  # type: ignore
        except Exception as e:
            # Checking for Snowflake token expiration by error code or message
            err_msg = str(e)
            err_code = getattr(e, "errno", None)
            if (
                ("Authentication token has expired" in err_msg)
                or (err_code == 390114)
                or ("390114" in err_msg)
            ):
                st.warning("Snowflake-Session abgelaufen. Stelle Verbindung wieder her...")

                # Clearing the cache to force a new session
                st.cache_resource.clear()
                snowflake_connection = create_session().connection
                try:
                    return snowflake_connection.cursor().execute(
                        "SHOW TABLES").fetchall()  # type: ignore
                except Exception as e2:
                    st.error(f"Fehler beim Abrufen der Tabellen nach Re-Login: {e2}")
                    return []
            else:
                st.error(f"Fehler beim Abrufen der Tabellen: {e}")
                return []
    tables = fetch_tables_with_retry()
    raw_tables = [
        row[1]
        for row in tables
        if row[1].startswith("LANGCHAIN")
    ]
    display_names = [
        name.removeprefix("LANGCHAIN_").title()
        for name in raw_tables
    ]
    name_map = dict(zip(display_names, raw_tables))
    options = ["Erstelle neue Tabelle"] + \
        display_names + ["Multi-Table-Selektion"]

    # Adding a selectbox for the user to select the table
    selected_disp = st.selectbox(
        "Wähle die Tabelle(n)",
        options,
        index=1,
        key="selected_table",
        on_change=_reset_vector_store
    )

    # Adding db table drop function it not "Erstelle neue Tabelle" or "Multi-Table-Selektion"
    if selected_disp not in ["Erstelle neue Tabelle", "Multi-Table-Selektion"]:
        # Adding a button to drop the table
        if st.button("Tabelle löschen", key="drop_table"):
            db_table_name = name_map[selected_disp]
            try:
                snowflake_connection.cursor().execute(
                    f"DROP TABLE IF EXISTS {db_table_name}").fetchall()
                st.success(f"Tabelle {db_table_name} wurde gelöscht!")
                del st.session_state["vector"]
                time.sleep(3)
                st.rerun()
            except Exception as e:
                st.error(f"Fehler beim Löschen der Tabelle: {e}")
    if selected_disp == "Erstelle neue Tabelle":
        new_disp = st.text_input(
            "Tabellenname", value="TEST", on_change=_reset_vector_store)
        st.session_state.option_embedding_model = st.selectbox("Wähle das Embedding-Modell", [
                                                               "multilingual-e5-large", "snowflake-arctic-embed-l-v2.0", "voyage-multilingual-2"], index=0, key="embedding_model")
        st.session_state.option_vector_length = st.selectbox(
            "Wähle die Vektorenlänge", [768, 1024], index=1, key="vector_length", disabled=True)
        if new_disp:
            table_name = new_disp if new_disp.startswith(
                "LANGCHAIN_") else f"LANGCHAIN_{new_disp}"
        else:
            table_name = "LANGCHAIN_TEST"
    else:
        if selected_disp == "Multi-Table-Selektion":
            # Adding a multiselect for the user to select multiple tables
            table_display_selection = st.multiselect(
                "Wähle mindestens 2 Tabellen",
                options=display_names,
                default=display_names[:2] if len(
                    display_names) >= 2 else display_names,
                key="multi_table_selection",
                on_change=_reset_vector_store
            )
            if len(table_display_selection) < 2:
                st.warning(
                    "Bitte wähle mindestens 2 Tabellen für die Multi-Table-Selektion.")
                table_name = []
            else:
                table_name = [name_map[n] for n in table_display_selection]
        else:
            table_name = name_map[selected_disp]
    st.session_state.option_table = table_name

    # Adding selectboxes for the user to select the LLM parameters
    if selected_disp != "Erstelle neue Tabelle":
        st.session_state.image_extraction = st.toggle("Bilder extrahieren", value=st.session_state.image_extraction)
        st.session_state.option_model = st.selectbox("Wähle ein Sprachmodell", [
                                                    "mistral-large2", "mistral-large", "mistral-7b", "llama3.1-70b", "llama3.1-8b"], index=0, key="model")

        # Setting system and assistant prompt
        system = st.text_input("System Instruktion", value=st.secrets["LLM"]["LLM_SYSTEM"], max_chars=500)
        system += st.text_input("System Instruktion+", value=f" {st.secrets['LLM']['LLM_SYSTEM_PLUS']}", max_chars=500)
        assistant = st.text_input("Assistant Instruktion", value=st.secrets["LLM"]["LLM_ASSISTANT"], max_chars=500)

# Showing the title
st.title(st.secrets["LLM"]["LLM_CHATBOT_NAME"])

# Creating form for user input for new table
if selected_disp == "Erstelle neue Tabelle":
    with st.form("vector_form"):
        # Setting the folder path to the parent directory of the current working dir
        folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

        # Getting the list of base folders and their subfolders (one and two levels deep)
        base_folders = [
            subfolder
            for subfolder in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, subfolder))
        ]
        subfolders = [
            os.path.join(subfolder, subsubfolder)
            for subfolder in base_folders
            for subsubfolder in os.listdir(os.path.join(folder, subfolder))
            if os.path.isdir(os.path.join(folder, subfolder, subsubfolder))
        ]
        options_offline_resources = base_folders + subfolders

        # Adding a selectbox for the user to select the folder
        st.session_state.option_offline_resources = st.selectbox(
            "Dokumente", options_offline_resources
        )

        # Adding a text input for the user to enter the URLs
        st.session_state.online_resources = st.text_area("URLs", disabled=True)

        # Adding the documents and a text input for the user to enter URLs
        if selected_disp == "Erstelle neue Tabelle":
            rag_perform_text = "Neue Inhalte integrieren"
        else:
            rag_perform_text = "Bestehende Inhalte verbinden"
        rag_perform = st.form_submit_button(rag_perform_text)
        if rag_perform:
            with st.spinner("Dokumente werden verarbeitet..."):
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

                            # Iterating over all files matched by the glob pattern
                            st.markdown("**Dokumente**")
                            for root, dirs, files in os.walk(self.directory_path):
                                for filename in files:
                                    for pattern in patterns:
                                        if fnmatch.fnmatch(filename, pattern):
                                            file_path = os.path.join(
                                                root, filename)

                                            # Checking if the file is a DOCX file
                                            if file_path.endswith(".docx"):
                                                loader = Docx2txtLoader(
                                                    file_path=file_path)

                                            # Checking if the file is a CSV file
                                            if file_path.endswith(".csv"):
                                                loader = CSVLoader(
                                                    file_path=file_path)

                                            # Checking if the file is a PDF file
                                            if file_path.endswith(".pdf"):
                                                pdf_file = fitz.open(file_path)

                                                # Iterating over PDF pages
                                                processed_hashes = set()
                                                for page_index in range(len(pdf_file)):
                                                    # Getting the page itself
                                                    page = pdf_file.load_page(
                                                        page_index)

                                                    # Getting images
                                                    if st.session_state.image_extraction:
                                                        image_list = page.get_images(
                                                            full=True)
                                                        for image_index, img in enumerate(image_list, start=1):
                                                            # Getting the XREF of the image
                                                            xref = img[0]

                                                            # Extracting the image bytes
                                                            base_image = pdf_file.extract_image(
                                                                xref)
                                                            image_bytes = base_image["image"]
                                                            image_ext = base_image["ext"]

                                                            # Filtering for images bigger
                                                            # 5 kB and smaller 1 MB
                                                            if len(image_bytes) > 10 * 1024 and len(image_bytes) < 1 * 1024 * 1024:
                                                                # Checking for duplicates
                                                                image_hash = hashlib.md5(
                                                                    image_bytes).hexdigest()
                                                                if image_hash in processed_hashes:
                                                                    continue
                                                                processed_hashes.add(
                                                                    image_hash)

                                                                # Saving the image
                                                                path = os.path.join(
                                                                    self.directory_path, "images")
                                                                os.makedirs(
                                                                    path, exist_ok=True)
                                                                image_name = f"image{page_index+1}_{image_index}.{image_ext}"
                                                                image_path = os.path.join(
                                                                    path, image_name)
                                                                with open(image_path, "wb") as image_file:
                                                                    image_file.write(
                                                                        image_bytes)
                                                loader = PyPDFLoader(
                                                    file_path=file_path)

                                            # Checking if the file is a text file
                                            if file_path.endswith(".txt"):
                                                # Ignoring `questions.txt` file
                                                if os.path.basename(file_path) == "questions.txt":
                                                    continue
                                                loader = TextLoader(
                                                    file_path=file_path)

                                            # Loading the file
                                            st.markdown(
                                                f"{os.path.basename(file_path)}")
                                            docs = loader.load()  # type: ignore
                                            for d in docs:
                                                # Adding page if not present
                                                if "page" not in d.metadata:
                                                    d.metadata["page"] = 0
                                                # Uploading file to MinIO and updating metadata['source']
                                                try:
                                                    minio_url = upload_file_to_minio(file_path)
                                                    d.metadata["source"] = minio_url
                                                except Exception as e:
                                                    st.warning(f"Fehler beim Hochladen nach MinIO: {e}")
                                            documents.extend(docs)

                            # Iterating over all URLs
                            st.markdown("**URLs**")
                            if len(self.urls[0]) > 0:
                                for url in self.urls:
                                    st.write(url.strip())
                                    loader = WebBaseLoader(url.strip(), requests_kwargs={"verify": False})
                                    docs = loader.load()
                                    for d in docs:
                                        if "page" not in d.metadata:
                                            d.metadata["page"] = 0
                                    documents.extend(docs)
                            return documents

                    # Setting the start time
                    st.session_state.start = time.time()
                    st.session_state.embeddings = SnowflakeEmbeddings(
                        connection=snowflake_connection, model=st.session_state.option_embedding_model
                    )

                    # Setting the folder path to the parent directory of the current
                    # working directory
                    folder = os.path.abspath(os.path.join(
                        os.getcwd(), '..', st.session_state.option_offline_resources))
                    urls = st.session_state.online_resources.split(',')
                    st.session_state.loader = CustomDirectoryLoader(
                        urls=urls, directory_path=folder, glob_pattern="*.docx|*.pdf|*.csv|*.txt")

                    # Loading the documents
                    st.session_state.docs = st.session_state.loader.load()

                    # Setting the configuration for the text splitter
                    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=100
                    )

                    # Splitting the documents into chunks
                    st.session_state.documents = st.session_state.text_splitter.split_documents(
                        st.session_state.docs
                    )

                    # Creating the vector store – explicitly specify your chosen table
                    st.session_state.vector = SnowflakeVectorStore.from_documents(
                        st.session_state.documents,
                        st.session_state.embeddings,
                        table=st.session_state.option_table,
                        vector_length=st.session_state.option_vector_length
                    )

                    # Showing the time taken
                    st.success(
                        f"Dokumente wurden in {int(time.time() - st.session_state.start)} Sekunden integriert!", icon="✅")
                    st.toast(
                        f"Dokumente wurden in {int(time.time() - st.session_state.start)} Sekunden integriert!", icon="✅")

                    # Waiting for 3 seconds and then reloading the page
                    time.sleep(3)
                    st.rerun()

# Connecting to existing vector store(s) if one or multi tables are selected (and not creating a new table)
if 'vector' not in st.session_state and selected_disp != "Erstelle neue Tabelle":
    st.session_state.embeddings = SnowflakeEmbeddings(
        connection=snowflake_connection,
        model=st.session_state.option_embedding_model
    )
    st.session_state.vector = SnowflakeVectorStore(
        connection=snowflake_connection,
        embedding=st.session_state.embeddings,
        table=st.session_state.option_table,
        vector_length=st.session_state.option_vector_length
    )

# Creating the chat interface
if 'vector' in st.session_state:
    if isinstance(st.session_state.option_table, list) and len(st.session_state.option_table) < 2:
        st.info("Bitte wähle mindestens 2 Tabellen für die Multi-Table-Abfrage.")
    else:
        # Setting up chat message history
        msgs = StreamlitChatMessageHistory(key="langchain_messages")
        if len(msgs.messages) == 0:
            msgs.add_ai_message(assistant)
        view_messages = st.expander("View the message contents in session state")

        # Preparing LLM and prompt with message history
        Cortex.model_rebuild()
        llm = Cortex(connection=snowflake_connection, model=st.session_state.option_model)
        prompt = ChatPromptTemplate(
            input_variables=["system", "history", "question", "context"],
            messages=[
                ("system", "{system}\n<context>\n{context}\n</context>"),
                ("human", st.secrets["LLM"]["LLM_USER_EXAMPLE"]),
                ("ai", st.secrets["LLM"]["LLM_ASSISTANT_EXAMPLE"]),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # Creating document & retrieval chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vector.as_retriever()

        # Ensuring output key for callbacks at retrieval_chain level
        retrieval_chain = create_retrieval_chain(retriever, document_chain) | RunnableLambda(ensure_output_key_chain)

        # Setting `RunnableWithMessageHistory` for chat
        chain_with_history = RunnableWithMessageHistory(
            retrieval_chain,
            lambda session_id: msgs,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Rendering chat history
        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        # Getting chat input
        user_input = st.chat_input("Frage stellen...")
        if user_input:
            st.chat_message("human").write(user_input)

            # Running chain with message history
            st.session_state.start = time.time()
            config = {"configurable": {"session_id": "any"}}
            input_data = {"system": system, "question": user_input, "context": "", "input": user_input}
            response = chain_with_history.invoke(input_data, config)

            # Storing the response in session state for downstream use (e.g., similarity search)
            st.session_state.response = response
            answer = None
            resp = st.session_state.response
            if isinstance(resp, dict):
                answer = resp.get("output")
                if answer is None:
                    # Trying 'answer' or first string value as fallback
                    answer = resp.get("answer")
                    if answer is None:
                        for v in resp.values():
                            if isinstance(v, str):
                                answer = v
                                break
            if answer is None:
                answer = str(resp)
            answer = answer.replace("Assistant: ", "").replace("\n", " ").lstrip()
            processing_time = int(time.time() - st.session_state.start)
            st.chat_message("ai").markdown(f"{answer} (verarbeitet in {processing_time} Sekunden)")

        # Showing similarity search results if available
        if (
            msgs.messages
            and msgs.messages[-1].type == "ai"
            and "response" in st.session_state
            and st.session_state.response
            and hasattr(st.session_state.response, "get")
            and st.session_state.response.get("context")
        ):
            with st.expander("Ähnlichkeitssuche"):
                for idx, doc in enumerate(st.session_state.response["context"]):
                    try:
                        tbl_name = doc.metadata.get("db_table")
                        st.write(f"**DB-Tabelle**: {tbl_name.replace('LANGCHAIN_', '').title()}")
                    except Exception:
                        if isinstance(st.session_state.option_table, list):
                            st.write(f"**DB-Tabelle**: {', '.join(st.session_state.option_table).replace('LANGCHAIN_', '').title()}")
                        else:
                            st.write(f"**DB-Tabelle**: {st.session_state.option_table.replace('LANGCHAIN_', '').title()}")
                    source = doc.metadata.get("source")
                    if source and isinstance(source, str) and source.startswith(("http://", "https://")):
                        filename = os.path.basename(source)
                        file_found = show_open_file_button(filename, source, idx)
                        if not file_found:
                            st.write(f"**Dateiname**: {filename}")
                    else:
                        # Fallback: just show the filename if not a valid URL
                        if source:
                            filename = os.path.basename(source)
                        else:
                            filename = "unbekannt"
                        st.write(f"**Dateiname**: {filename}")
                    st.write("**Inhalt:**")
                    st.text(doc.page_content)
                    st.write("---------------------------")

        # Drawing the messages at the end, so newly generated ones show up immediately
        with view_messages:
            # Creating a function to serialize messages
            def serialize_message(msg):
                return {
                    "type": getattr(msg, 'type', type(msg).__name__),
                    "content": getattr(msg, 'content', str(msg)),
                    **{k: v for k, v in msg.__dict__.items() if k not in ('type', 'content')}
                }
            messages_json = [serialize_message(m) for m in msgs.messages]

            # Adding a download button for the message history
            if st.session_state["IS_EMBED"]:
                # Showing the copy-to-clipboard in Angular app/iframe
                st.code(json.dumps(messages_json, ensure_ascii=False, indent=2), language="json")
            else:
                # Showing the download button in normal browser
                st.json(messages_json)
                st.markdown("**Chatverlauf als JSON downloaden:**")
                st.download_button(
                    label="Download Chatverlauf (JSON)",
                    data=json.dumps(messages_json, ensure_ascii=False, indent=2),
                    file_name="chat_history.json",
                    mime="application/json"
                )
else:
    if selected_disp == "Erstelle neue Tabelle":
        st.info("Bitte integriere zuerst Dokumente, um eine Vektorbank zu erstellen.")

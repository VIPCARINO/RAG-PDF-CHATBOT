import streamlit as st
import tempfile
import os
import uuid
import time
import json
from database import MetadataDB
from helper import delete_from_chroma

from ingestion_pipeline_class_based import PDFIngestionPipeline
from RAG_Phase_5_query_answer_class_based import LocalRAGPipeline

api_key = st.secrets["api_key"]

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PDF RAG Assistant",
    layout="wide"
)

st.title("📚 Chat With Your PDF")


# =========================================================
# STORAGE FILES
# =========================================================
CHAT_HISTORY_FILE = "./json_files/chat_histories.json"


# =========================================================
# DATABASE
# =========================================================
db = MetadataDB()


# =========================================================
# CHAT HISTORY STORAGE
# =========================================================
def load_chat_histories():

    if not os.path.exists(CHAT_HISTORY_FILE):
        return {}

    try:

        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return {}


def save_chat_histories(histories):

    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)


# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = None

# =========================================================
# STATUS UPDATE
# =========================================================
def create_stage_ui(title):

    container = st.container()

    progress_bar = container.progress(0)

    status_box = container.empty()

    start_time = time.time()

    def callback(data):

        progress = float(data.get("progress", 0))

        percent = int(progress * 100)

        processed = data.get("processed", 0)

        total = data.get("total", 1)

        speed = data.get("speed", 0)

        eta = data.get("eta", None)

        message = data.get("message", "")

        # ==================================================
        # COMPLETED STAGE
        # ==================================================
        if progress >= 1.0:

            progress_bar.empty()

            status_box.empty()

            status_box.success(f"✅ {title} Complete")

            return

        # ==================================================
        # ACTIVE STAGE
        # ==================================================
        progress_bar.progress(percent)

        status_box.markdown(f"""
### {title}

- **Progress:** {processed}/{total} ({percent}%)
- **Speed:** {speed:.2f} items/sec
- **ETA:** {eta if eta else 'Calculating...'} sec
- **Message:** {message}
""")

    return callback

# =========================================================
# DELETE CHAT HISTORY
# =========================================================
def delete_chat_history(chat_histories, doc_id, save_func):
    if doc_id in chat_histories:
        del chat_histories[doc_id]
    save_func(chat_histories)

# =========================================================
# LOAD ALL CHAT HISTORIES
# =========================================================
chat_histories = load_chat_histories()


# =========================================================
# BUILD SHORT MEMORY
# =========================================================
def build_chat_history(messages, max_messages=6):

    recent = messages[-max_messages:]

    history = []

    for msg in recent:

        role = msg["role"].upper()

        history.append(
            f"{role}: {msg['content']}"
        )

    return "\n".join(history)


# =========================================================
# LOAD DOCUMENT CHAT
# =========================================================
def load_document_chat(doc_id, filename):

    st.session_state.doc_id = doc_id
    st.session_state.active_pdf = filename
    st.session_state.pdf_ready = True

    if doc_id in chat_histories:
        st.session_state.messages = chat_histories[doc_id]
    else:
        st.session_state.messages = []


# =========================================================
# SAVE DOCUMENT CHAT
# =========================================================
def save_document_chat(doc_id, messages):

    chat_histories[doc_id] = messages

    save_chat_histories(chat_histories)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    process_btn = st.button(
        "⚡ Process PDF",
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # DOCUMENT LIBRARY
    # =====================================================
    st.subheader("📚 Available PDFs")

    documents = db.get_all_documents()

    shown_files = set()

    for doc in documents:

        doc_id = doc[0]
        filename = doc[1]

        # ---------------------------------------------
        # SHOW ONLY UNIQUE FILES
        # ---------------------------------------------
        if filename in shown_files:
            continue

        shown_files.add(filename)

        col1, col2 = st.columns([5, 1])

        with col1:

            # ---------------------------------------------
            # ACTIVE STYLE
            # ---------------------------------------------
            button_type = (
                "primary"
                if st.session_state.doc_id == doc_id
                else "secondary"
            )

            if st.button(
                f"📘 {filename}",
                key=f"open_{doc_id}",
                type=button_type
            ):
                load_document_chat(doc_id, filename)
                st.rerun()
        # =====================================================
        # DELETE DOCUMENT
        # =====================================================
        with col2:

            if st.button(
                "🗑️",
                key=f"delete_{doc_id}"
            ):

                # ================================
                # DELETE FROM SQLITE
                # ================================
                db.delete_document(doc_id)

                # ================================
                # DELETE FROM CHROMA
                # ================================
                delete_from_chroma(
                    db_path="./database/chroma_db",
                    collection_name="pdf_rag",
                    doc_id=doc_id
                )

                # ================================
                # DELETE CHAT HISTORY
                # ================================
                chat_histories = load_chat_histories()

                if doc_id in chat_histories:
                    del chat_histories[doc_id]
                    save_chat_histories(chat_histories)

                # ================================
                # RESET UI IF ACTIVE
                # ================================
                if st.session_state.doc_id == doc_id:
                    st.session_state.doc_id = None
                    st.session_state.active_pdf = None
                    st.session_state.messages = []
                    st.session_state.pdf_ready = False

                st.success(f"Deleted {filename}")
                st.rerun()

    st.divider()

    # =====================================================
    # ACTIVE DOCUMENT
    # =====================================================
    st.subheader("📑 Active Document")

    if st.session_state.active_pdf:

        st.success(
            st.session_state.active_pdf
        )

    else:

        st.info("No document selected")

    st.divider()

    # =====================================================
    # CLEAR CHAT
    # =====================================================
    if st.session_state.doc_id:

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True
        ):

            st.session_state.messages = []

            save_document_chat(
                st.session_state.doc_id,
                []
            )

            st.rerun()


# =========================================================
# MAIN LAYOUT
# =========================================================
left_col, right_col = st.columns([3, 1])


# =========================================================
# RIGHT PANEL
# =========================================================
with right_col:

    st.subheader("📘 Current PDF")

    if st.session_state.active_pdf:

        st.markdown(
            f"""
### ✅ Loaded

**File:**  
`{st.session_state.active_pdf}`

**Doc ID:**  
`{st.session_state.doc_id}`

**Messages:**  
`{len(st.session_state.messages)}`
"""
        )

    else:

        st.info(
            "Upload or select a PDF from the sidebar."
        )


# =========================================================
# PROCESS PDF (CLEAN VERSION - NO CALLBACKS)
# =========================================================
if uploaded_file and process_btn:

    st.session_state.pdf_ready = False

    try:

        # ---------------------------------------------
        # SAVE TEMP FILE
        # ---------------------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # ---------------------------------------------
        # DOC ID
        # ---------------------------------------------
        doc_id = str(uuid.uuid4())

        # ---------------------------------------------
        # CHECK IF FILE EXISTS
        # ---------------------------------------------
        existing_doc = db.document_exists_by_name(uploaded_file.name)

        if existing_doc:

            existing_doc_id, existing_filename = existing_doc

            st.warning("⚠️ This PDF already exists in the database.")
            st.info(f"Loading existing document: {existing_filename}")

            st.session_state.doc_id = existing_doc_id
            st.session_state.active_pdf = existing_filename
            st.session_state.pdf_ready = True

            if existing_doc_id in chat_histories:
                st.session_state.messages = chat_histories[existing_doc_id]
            else:
                st.session_state.messages = []

            st.stop()

        # ---------------------------------------------
        # SIMPLE INGESTION (NO CALLBACK)
        # ---------------------------------------------
        with st.spinner("Processing PDF... This may take a moment."):

            with st.container():

                # =====================================================
                # 1. PDF INGESTION STAGES WRAPPER
                # =====================================================

                extract_cb = create_stage_ui("📄 PDF Extraction")
                structure_cb = create_stage_ui("🧩 Structure Parsing")
                chunk_cb = create_stage_ui("✂️ Chunking")
                embed_cb = create_stage_ui("🧠 Embedding")
                db_cb = create_stage_ui("💾 Database Indexing")
                # =====================================================
                # PIPELINE RUNNER
                # =====================================================

                with st.spinner("Running full RAG pipeline..."):

                    pipeline = PDFIngestionPipeline(
                        pdf_path=pdf_path,
                        collection_name="pdf_rag",
                        doc_id=doc_id,
                        filename=uploaded_file.name,

                        # 🔥 PASS CALLBACKS HERE
                        extract_callback=extract_cb,
                        structure_callback=structure_cb,
                        chunk_callback=chunk_cb,
                        embed_callback=embed_cb,
                        db_callback=db_cb
                    )

                    result = pipeline.run()

        # ---------------------------------------------
        # SESSION STATE
        # ---------------------------------------------
        st.session_state.doc_id = doc_id
        st.session_state.active_pdf = uploaded_file.name
        st.session_state.pdf_ready = True
        st.session_state.messages = []

        save_document_chat(doc_id, [])

        # ---------------------------------------------
        # CLEANUP
        # ---------------------------------------------
        os.unlink(pdf_path)

        st.success(f"📚 {uploaded_file.name} ready for chatting.")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

with left_col:
    # =========================================================
    # CHAT HISTORY
    # =========================================================

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])


# =========================================================
# CHAT INPUT
# =========================================================
if st.session_state.pdf_ready:

    prompt = st.chat_input(
        "Ask your PDF something..."
    )

    if prompt:

        # =================================================
        # SAVE USER MESSAGE
        # =================================================
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with left_col:

            # ---------------------------------------------
            # USER CHAT
            # ---------------------------------------------
            with st.chat_message("user"):

                st.markdown(prompt)

            # ---------------------------------------------
            # LOAD RAG
            # ---------------------------------------------
            rag = LocalRAGPipeline(
                api_key=api_key,
                collection_name="pdf_rag",
                doc_id=st.session_state.doc_id
            )

            # ---------------------------------------------
            # ASSISTANT CHAT
            # ---------------------------------------------
            #with st.chat_message("assistant"):

                #with st.spinner("Thinking..."):

                    # =====================================
                    # RAG SEARCH
                    # =====================================
                    #answer = rag.run(prompt)

                    #st.markdown(answer)

            # =================================================
            # STEAM ASSISTANT CHAT
            # =================================================
            with st.chat_message("assistant"):
                placeholder = st.empty()

                full_response = ""

                with st.spinner("Thinking..."):
                    stream = rag.run(prompt)

                    for token in stream:
                        full_response += token
                        placeholder.markdown(full_response)

        # =================================================
        # SAVE ASSISTANT MESSAGE
        # =================================================
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
            #"role": answer, # No Streaming
        })

        # =================================================
        # SAVE DOCUMENT CHAT
        # =================================================
        save_document_chat(
            st.session_state.doc_id,
            st.session_state.messages
        )

else:

    with left_col:

        st.info(
            "👈 Upload or select a PDF from the sidebar."
        )

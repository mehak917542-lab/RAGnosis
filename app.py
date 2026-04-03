import streamlit as st
import os, sys, tempfile
from ingest import load_document, chunk_documents, store_in_chromadb, get_vectorstore
from kg_builder import build_kg
from chain import RAGChain

# Page config 
st.set_page_config(
    page_title="KG-RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 KG-RAG Chatbot")
st.caption("Upload a PDF → Build Knowledge Graph → Chat with your document")

# Sidebar 
with st.sidebar:
    st.header("Upload Document")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        # Save uploaded file to /data directory
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Saved: {uploaded_file.name}")

        # Process button
        if st.button("Process PDF", type="primary", use_container_width=True):
            # Clear old chat + chain
            st.session_state.messages = []
            st.session_state.chain_ready = False
            st.session_state.pop("chain", None)

            # Step 1: Ingest → ChromaDB
            with st.status("Ingesting document...", expanded=True) as status:
                try:
                    st.write("Loading PDF pages...")
                    docs = load_document(uploaded_file.name)
                    st.write(f"{len(docs)} pages loaded")

                    st.write("Chunking text...")
                    chunks = chunk_documents(docs)
                    st.write(f"{len(chunks)} chunks created")

                    st.write("Embedding + saving to ChromaDB...")
                    store_in_chromadb(chunks)
                    st.write("ChromaDB ready")

                    # Step 2: Build KG → Neo4j
                    st.write("Extracting triples → Neo4j Knowledge Graph...")
                    build_kg(uploaded_file.name)
                    st.write("Knowledge Graph built")

                    status.update(label="PDF processed! You can now chat.", state="complete")
                    st.session_state.chain_ready = True
                    st.session_state.current_doc = uploaded_file.name

                except Exception as e:
                    status.update(label="Processing failed", state="error")
                    st.error(str(e))

    st.divider()

    # Settings
    st.header("Settings")
    show_kg     = st.toggle("Show KG facts", value=True)
    show_chunks = st.toggle("Show source chunks", value=False)

    st.divider()
    st.markdown("""
    **How it works:**
    1. Upload + Process your PDF
    2. **ChromaDB** stores semantic chunks
    3. **Neo4j** stores entity relationships  
    4. **LLaMA via Groq** answers your questions
    """)

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Load RAG chain (only after PDF is processed) 
@st.cache_resource
def load_chain():
    return RAGChain()


# Main area 
if not st.session_state.get("chain_ready"):
    # Show a friendly landing state
    st.info("Upload a PDF from the sidebar and click **Process PDF** to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Step 1\nUpload any PDF document from the sidebar")
    with col2:
        st.markdown("### Step 2\nClick **Process PDF** to build the knowledge graph")
    with col3:
        st.markdown("### Step 3\nAsk questions — get answers with KG + RAG context")

else:
    # Show which doc is loaded
    st.success(f"Active document: **{st.session_state.get('current_doc', 'Unknown')}**")

    # Load chain (cached)
    try:
        if "chain" not in st.session_state:
            with st.spinner("Loading RAG chain..."):
                st.session_state.chain = load_chain()
        chain = st.session_state.chain
    except Exception as e:
        st.error(f"Failed to load chain: {e}")
        st.stop()

    # ── Chat history ──────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("kg_facts") and show_kg:
                with st.expander("Knowledge Graph Facts"):
                    st.code(msg["kg_facts"])
            if msg.get("sources") and show_chunks:
                with st.expander("Source Chunks"):
                    for s in msg["sources"]:
                        st.markdown(f"**Page {s['page']}:** {s['snippet']}...")

    # Chat input 
    if question := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving from ChromaDB + Neo4j → asking LLM..."):
                try:
                    result = chain.ask(question)
                    answer = result["answer"]
                    st.markdown(answer)

                    if result["kg_facts"] and show_kg:
                        with st.expander("Knowledge Graph Facts used"):
                            st.code(result["kg_facts"])

                    if show_chunks:
                        with st.expander("Source Chunks"):
                            for s in result["sources"]:
                                st.markdown(f"**Page {s['page']}:** {s['snippet']}...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "kg_facts": result["kg_facts"],
                        "sources": result["sources"]
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
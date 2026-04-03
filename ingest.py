import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Config 
DATA_DIR       = "data"
CHROMA_DB_DIR  = "chroma_db"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"  

# Step 1: Load PDF
def load_document(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}. Drop your PDF inside the /data folder.")
    
    print(f"Loading document: {filename}")
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    return docs

# Step 2: Chunk the document 
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # characters per chunks
        chunk_overlap=50,      # overlap so context isn't lost at boundaries
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


# Step 3: Embed + Store in ChromaDB
def store_in_chromadb(chunks):
    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print(f"Storing chunks in ChromaDB at ./{CHROMA_DB_DIR} ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print(f"ChromaDB ready — {len(chunks)} chunks stored")
    return vectorstore   
 
 
# Step 4: Load existing ChromaDB  
def get_vectorstore():
    """Load an already-persisted ChromaDB (used by retriever.py)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

# Step 4: Test retrieval 
def test_retrieval(vectorstore, query: str = "What is this document about?"):
    print(f"\nTest query: '{query}'")
    results = vectorstore.similarity_search(query, k=3)
    for i, r in enumerate(results):
        print(f"\n--- Chunk {i+1} (page {r.metadata.get('page', '?')}) ---")
        print(r.page_content[:300])

# Main 
if __name__ == "__main__":
    import sys

    # Pass filename as argument: python ingest.py mypaper.pdf
    # Or default to first PDF found in /data
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        if not pdfs:
            print("No PDF found in /data. Add one and retry.")
            sys.exit(1)
        filename = pdfs[0]
        print(f"Auto-detected: {filename}")

    docs   = load_document(filename)
    chunks = chunk_documents(docs)
    vs     = store_in_chromadb(chunks)
    test_retrieval(vs)
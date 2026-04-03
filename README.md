KG-RAG Chatbot

A Knowledge Graph + Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload a PDF and interact with it using natural language.

It combines:

📄Semantic Search (ChromaDB)
🔗 Knowledge Graph (Neo4j)
🤖 LLM (LLaMA via Groq API)

🚀 Features
Upload any PDF document
Automatically:
Split into chunks
Generate embeddings
Build Knowledge Graph
Ask questions and get:
Context-aware answers
Source references
Knowledge Graph facts

🏗️ Architecture
PDF → Chunking → Embeddings → ChromaDB
                     ↓
               Knowledge Graph (Neo4j)
                     ↓
        Hybrid Retriever (Vector + KG)
                     ↓
                LLM (Groq)
                     ↓
                  Answer

📂 Project Structure
├── app.py              # Streamlit UI
├── chain.py            # RAG pipeline (LLM + retrieval)
├── retriever.py        # Hybrid retriever (ChromaDB + Neo4j)
├── ingest.py           # PDF processing & vector storage
├── kg_builder.py       # Knowledge Graph creation
├── data/               # Uploaded PDFs
├── chroma_db/          # Vector database
├── requirements.txt
└── .env                # API keys (not to be shared)

⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/your-username/kg-rag-chatbot.git
cd kg-rag-chatbot
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Create .env file
HUGGINGFACEHUB_API_TOKEN=your_token
NEO4J_URI=your_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
GROQ_API=your_api_key
4️⃣ Run Neo4j
Use Neo4j Aura (cloud) OR local Neo4j instance
5️⃣ Run the app
streamlit run app.py
💡 How It Works
🔹 Step 1: Document Ingestion
PDF is loaded using PyMuPDF
Split into chunks
Stored in ChromaDB
🔹 Step 2: Knowledge Graph
LLM extracts (subject, relation, object) triples
Stored in Neo4j
🔹 Step 3: Hybrid Retrieval
Semantic search → ChromaDB
Entity-based search → Neo4j
🔹 Step 4: Answer Generation
Combined context → LLaMA model via Groq
Returns:
Answer
Sources
KG facts
🧪 Example Query
"What are the key findings of the paper?"

Output includes:

✅ Answer
📊 Knowledge Graph facts
📄 Source references
🛠️ Tech Stack
LangChain
ChromaDB
Neo4j
Groq (LLaMA 3.1)
HuggingFace Embeddings
Streamlit
⚠️ Important Notes
❌ Do NOT commit .env file
❌ Do NOT expose API keys
✔️ Add .env to .gitignore
📌 Future Improvements
Add multi-document support
Improve KG extraction accuracy
Add chat memory
Deploy on cloud (AWS / GCP)
🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first.


👨‍💻 Author
Mehak 

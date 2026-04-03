import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from retriever import HybridRetriever
from groq import Groq

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL = "llama-3.1-8b-instant"


# Prompt builder 
# Constructs the final prompt sent to Mistral LLM. Uses the [INST] instruction format 
# specific to Mistral-Instruct models. Injects the retrieved context (vector + graph)
# and the user question, with strict instructions to only use provided context.
def build_prompt(context: str, question: str) -> str:
    return f"""
            You are a helpful research assistant. Answer the question using ONLY the context provided.
            If the answer is not in the context, say "I don't have enough information to answer that."
            Be concise and factual.

            Context:
            {context}

            Question: {question}
            """


# Main QA chain 
# Main class that runs the complete RAG pipeline:
# 1. Retrieves hybrid context (ChromaDB vectors + Neo4j graph facts)
# 2. Sends to llama-3.1 LLM via HuggingFace API
# 3. Returns structured answer with source attribution
class RAGChain:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.client = Groq(api_key=os.getenv("GROQ_API"))
        print("Mistral chain ready")

    def ask(self, question: str, verbose: bool = False) -> dict:
        """
        Full RAG pipeline:
        question → hybrid retrieval → Mistral → answer

        Returns dict with answer + sources for transparency.
        """
        
        # Step 1: Retrieve
        retrieval = self.retriever.retrieve(question, k=4)
        context   = retrieval["combined_context"]

        if verbose:
            print("\n Retrieved Context ")
            print(context[:800], "..." if len(context) > 800 else "")

        # Step 2: Generate
        prompt = build_prompt(context, question)
        
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=HF_MODEL,
            temperature=0.2,
            max_tokens=300
        )

        # Extract text from chat response
        response_text = response.choices[0].message.content
        
        # Clean up response (remove any trailing artifacts)
        answer = response_text.strip()

        return {
            "question": question,
            "answer": answer,
            "kg_facts": retrieval["kg_facts"],
            "sources": [
                {
                    "page": d.metadata.get("page", "?"),
                    "snippet": d.page_content[:150]
                }
                for d in retrieval["semantic_chunks"]
            ]
        }

    def close(self):
        self.retriever.close()


# CLI test 
if __name__ == "__main__":
    chain = RAGChain()
    print("\n KG-RAG Chatbot (type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            break
        if not question:
            continue

        result = chain.ask(question, verbose=False)
        print(f"\n🤖 Answer:\n{result['answer']}")

        if result["kg_facts"]:
            print(f"\n📊 KG Facts used:\n{result['kg_facts']}")

        print(f"\n📄 Sources: pages {[s['page'] for s in result['sources']]}")
        print("─" * 50)

    chain.close()
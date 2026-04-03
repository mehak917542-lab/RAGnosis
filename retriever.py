import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from ingest import get_vectorstore

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# Neo4j retrieval 
# Connects to Neo4j and retrieves structured knowledge (triples) based on 
# entity keywords extracted from the user query. Returns formatted facts.
class Neo4jRetriever:
    def __init__(self):
        
        # Establishes connection to Neo4j
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

    # Cleanup utility: closes the database connection when done
    def close(self):
        self.driver.close()

    # Core retrieval logic: searches Knowledge Graph for triples where either
    # Subject or Object contains any of the provided keywords.
    def query(self, entity_keywords: list[str]) -> str:
        """
        Given a list of keywords, find related triples in the graph.
        Returns a formatted string of facts.
        """
        facts = []
        query = """ 
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE ANY(kw IN $keywords WHERE
                        toLower(a.name) CONTAINS toLower(kw) OR
                        toLower(b.name) CONTAINS toLower(kw))
                    RETURN a.name AS subject, r.type AS relation, b.name AS object
                    LIMIT 15
                """
        with self.driver.session() as s:
            results = s.run(query, keywords=entity_keywords)
            for rec in results:
                facts.append(f"{rec['subject']} → {rec['relation']} → {rec['object']}")

        if facts:
            return "Knowledge Graph Facts:\n" + "\n".join(facts)
        return ""


# Keyword extractor (simple, no extra model needed)
def extract_keywords(query: str) -> list[str]:
    """
    Naive keyword extractor : filters stopwords, returns meaningful tokens.
    Good enough for KG lookup without needing spaCy/NER.
    """
    stopwords = {
        "what","is","are","the","a","an","of","in","on","at","to","for",
        "how","why","who","when","where","does","do","was","were","has",
        "have","had","be","been","being","and","or","but","with","from",
        "this","that","these","those","it","its","their","there","about",
        "can","could","would","should","will","tell","me","explain","give"
    }
    tokens = query.lower().split()
    keywords = [t.strip("?.!,") for t in tokens if t not in stopwords and len(t) > 2]
    return keywords


# Hybrid Retriever 
# Combines both Vector Search (ChromaDB for semantic similarity) and 
# Graph Search (Neo4j for logical connections) to provide comprehensive 
# context to the LLM. This is the main retrieval engine of your RAG system
class HybridRetriever:
    def __init__(self):
        print("Loading ChromaDB vectorstore...")
        self.vectorstore   = get_vectorstore()
        self.neo4j         = Neo4jRetriever()
        print("Hybrid retriever ready")

    def retrieve(self, query: str, k: int = 4) -> dict:
        """
        Returns:
            {
                "semantic_chunks": [...],   # from ChromaDB
                "kg_facts": "...",          # from Neo4j
                "combined_context": "..."   # merged string for LLM
            }
        """
        
        # 1. Semantic retrieval from ChromaDB
        semantic_docs = self.vectorstore.similarity_search(query, k=k)
        semantic_text = "\n\n".join([d.page_content for d in semantic_docs])

        # 2. KG retrieval from Neo4j
        keywords = extract_keywords(query)
        kg_facts = self.neo4j.query(keywords)

        # 3. Combine
        combined = ""
        if kg_facts:
            combined += f"{kg_facts}\n\n"
        combined += f"Document Excerpts:\n{semantic_text}"

        return {
            "semantic_chunks": semantic_docs,
            "kg_facts": kg_facts,
            "combined_context": combined
        }

    def close(self):
        self.neo4j.close()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = HybridRetriever()
    query = input("Enter a test query: ")
    result = retriever.retrieve(query)

    print("\n── KG Facts ──────────────────────────────")
    print(result["kg_facts"] or "(none found)")
    print("\n── Semantic Chunks ───────────────────────")
    for i, doc in enumerate(result["semantic_chunks"]):
        print(f"\nChunk {i+1}: {doc.page_content[:200]}")
    retriever.close()
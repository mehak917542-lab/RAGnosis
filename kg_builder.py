import os, re, json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from ingest import load_document, chunk_documents
from groq import Groq

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
HF_TOKEN       = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROK_API       = os.getenv("GROQ_API")

# use llama-3.1-8b-instant via GROQ API
HF_MODEL = "llama-3.1-8b-instant",


# knowldege graph builder
class KnowledgeGraph:
    def __init__(self):
        
        # establish a connection with neo4j database and create a driver object
        # the driver manage all communication between python code and graph database
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        print("Connected to Neo4j")

    
    # utility to close the connection. 
    def close(self):
        self.driver.close()

    # utility to close/reset the session [clear graph entities and relations]
    def clear(self):
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        print("Graph cleared")

    # insert a triple to the graph Knowledge graph (KG)
    # two entities and 1 relation. 
    # Uses MERGE to avoid duplicates (creates only if doesn't exist).
    def insert_triple(self, subject: str, relation: str, obj: str):
        query = """
        MERGE (a:Entity {name: $subject})
        MERGE (b:Entity {name: $obj})
        MERGE (a)-[r:RELATION {type: $relation}]->(b)
        """
        with self.driver.session() as s:
            s.run(query, subject=subject.strip(), relation=relation.strip(), obj=obj.strip())

    # Searches the knowledge graph for any triples connected to a given entity.
    # Returns up to 10 matching (subject, relation, object) triples.
    def query_entity(self, entity: str) -> list[dict]:
        query = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE toLower(a.name) CONTAINS toLower($entity)
           OR toLower(b.name) CONTAINS toLower($entity)
        RETURN a.name AS subject, r.type AS relation, b.name AS object
        LIMIT 10
        """
        with self.driver.session() as s:
            result = s.run(query, entity=entity)
            return [dict(record) for record in result]


# # Extract triples from text using Mistral 7B LLM, returns JSON format, max 8 triples
def extract_triples(text: str, client: Groq) -> list[tuple]:
    prompt = f"""Extract factual (subject, relation, object) triples from the text below.
                Return ONLY a JSON array like: [{{"subject":"X","relation":"Y","object":"Z"}}]
                Do not add explanation. Max 8 triples.

                Text:
                {text}

                JSON:"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=300
    )

    # Extract text from chat response
    response_text = chat_completion.choices[0].message.content
    
    
    # Parse JSON from response
    try:
        # Find JSON array in the response
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            triples_raw = json.loads(match.group())
            return [(t["subject"], t["relation"], t["object"]) for t in triples_raw
                    if all(k in t for k in ["subject", "relation", "object"])]
    except Exception as e:
        print(f"Parse error: {e}")
    return []


# ── Main ──────────────────────────────────────────────────────────────────────
def build_kg(filename: str):
    import sys, os
    DATA_DIR = "data"

    docs   = load_document(filename)
    chunks = chunk_documents(docs)

    # Only process first 20 chunks 
    chunks = chunks[:20]
    print(f"\n Extracting triples from {len(chunks)} chunks via Mistral")

    client = Groq(api_key=GROK_API) 
    kg     = KnowledgeGraph()
    kg.clear()

    total_triples = 0
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}/{len(chunks)} ...", end=" ")
        triples = extract_triples(chunk.page_content, client)
        for s, r, o in triples:
            kg.insert_triple(s, r, o)
        total_triples += len(triples)
        print(f"{len(triples)} triples")

    print(f"\n Knowledge Graph built — {total_triples} triples stored in Neo4j")
    kg.close()


if __name__ == "__main__":
    import sys, os

    DATA_DIR = "data"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        if not pdfs:
            print(" No PDF in /data.")
            sys.exit(1)
        filename = pdfs[0]

    build_kg(filename)
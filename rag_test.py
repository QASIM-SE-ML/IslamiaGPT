import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "islamia_vector_db")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

collection = client.get_collection("islamia_knowledge")

question = input("Ask a question: ")

query_embedding = embedding_model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print("\n--- Retrieved Information ---\n")

THRESHOLD = 1.0

found = False

for i, document in enumerate(results["documents"][0]):

    distance = results["distances"][0][i]

    if distance < THRESHOLD:
        found = True

        print(f"Result {i + 1}:")
        print(document)
        print("Source:", results["metadatas"][0][i]["source"])
        print("Distance:", distance)
        print()

if not found:
    print("No sufficiently relevant information found.")
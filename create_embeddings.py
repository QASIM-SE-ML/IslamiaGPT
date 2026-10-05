import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "islamia_vector_db")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

collection = client.get_or_create_collection(
    name="islamia_knowledge"
)

for filename in os.listdir(KNOWLEDGE_DIR):

    if filename.endswith(".txt"):

        filepath = os.path.join(KNOWLEDGE_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        # Split by paragraphs first
        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []

        for paragraph in paragraphs:

            if len(paragraph) <= 700:
                chunks.append(paragraph)

            else:
                # Split long paragraphs into overlapping chunks
                for i in range(0, len(paragraph), 600):
                    chunk = paragraph[i:i + 700]
                    chunks.append(chunk)

        for index, chunk in enumerate(chunks):

            embedding = embedding_model.encode(chunk).tolist()

            collection.upsert(
                ids=[f"{filename}_{index}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": filename}]
            )

print("Knowledge successfully added to vector database!")
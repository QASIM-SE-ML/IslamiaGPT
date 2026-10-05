import chromadb

client = chromadb.PersistentClient(path="test_db")

collection = client.get_or_create_collection("test")

collection.add(
    ids=["1"],
    documents=["Islamia College Peshawar"],
)

print("ChromaDB is working!")
import chromadb
import json
import os

# Create ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.get_or_create_collection(
    name="indala_admission"
)

# Load embeddings
with open("embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Store vectors
collection.add(
    ids=[str(i) for i in range(len(data["chunks"]))],
    documents=data["chunks"],
    embeddings=data["embeddings"]
)

print("Embeddings stored successfully!")
print("Total documents:", collection.count())
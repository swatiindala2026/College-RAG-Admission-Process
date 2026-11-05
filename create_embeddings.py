import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load chunks
with open("outputs/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
documents = [chunk["text"] for chunk in chunks]
embeddings = model.encode(documents).tolist()

# Store in ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="college_admission"
)

collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=documents,
    metadatas=[
        {"source": chunk["source"], "page": chunk["page"]}
        for chunk in chunks
    ],
    embeddings=embeddings
)

print("Embeddings created successfully!")
print("Total documents stored:", collection.count())
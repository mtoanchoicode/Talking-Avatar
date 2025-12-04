import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables (optional)
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "chat-memory"

# Create index if it doesn't exist
# if index_name not in [idx.name for idx in pc.list_indexes()]:
#     pc.create_index(
#         name=index_name,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
#     print(f"Created new Pinecone index: {index_name}")

# Connect to the index
index = pc.Index(index_name)
print(f"Connected to Pinecone index: {index_name}")

# ---------- DELETE ALL RECORDS ----------
print("Deleting all vectors from index...")
index.delete(delete_all=True)
print("All vectors deleted successfully.")

# Optionally verify index stats
stats = index.describe_index_stats()
print("Index stats after deletion:", stats)

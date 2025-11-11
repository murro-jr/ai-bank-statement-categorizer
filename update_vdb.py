from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#connect to QDrant Vector DB
qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))

collection_name = "transactions"
if (qdrant.collection_exists(collection_name=collection_name) != True):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

def embed_text(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# transactions=[]
# points = []
# for tx in transactions:
#     vector = embed_text(tx["desc"])
#     points.append({
#         "id": tx["id"],
#         "vector": vector,
#         "payload": {
#             "description": tx["desc"],
#             "category": tx["category"]
#         }
#     })

# qdrant.upsert(collection_name=collection_name, points=points)
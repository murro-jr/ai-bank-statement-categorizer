from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from openai import OpenAI

import pandas as pd

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#connect to QDrant Vector DB
qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))

collection_name = "transactions"
qdrant.delete_collection(collection_name=collection_name)
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

df = pd.read_csv(os.getenv('DATAFILE_PATH'))
transactions = df.to_dict(orient="records")
points = []
for tx in transactions:
    vector = embed_text(tx["description"])
    points.append({
        "id": tx["id"],
        "vector": vector,
        "payload": {
            "description": tx["description"],
            "category": tx["category"],
            "amount": tx["amount"]
        }
    })

qdrant.upsert(collection_name=collection_name, points=points)
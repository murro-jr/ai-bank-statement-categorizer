import sys
from dotenv import load_dotenv
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from openai import OpenAI

from gemini_service import extract_and_categorize_transactions

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#connect to QDrant Vector DB
qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))

def embed_text(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def add_point_to_qdrant(id, trx, vector):
    point_to_update = {
        "id": id,
        "vector": vector,
        "payload": {
            "description": trx["description"],
            "category": trx["category"],
            "amount": trx["amount"]
        }
    }
    qdrant.upsert(collection_name="transactions", points=[point_to_update])

if len(sys.argv) > 2:
    filename = sys.argv[1]
    run_upsert = sys.argv[2]
    print('RUN UPSERT: ', run_upsert)
    transactions = extract_and_categorize_transactions(file_path=filename, mime_type='application/pdf')
    for trx in transactions:
        vector = embed_text(trx['description'])
        search_results = qdrant.query_points(
            collection_name="transactions",
            query=vector,  # same vector as before
            limit=1
        )
        data = search_results.points[0]
        if not int(run_upsert):
            print(trx)
            print('SCORE: ', data.score, ', PAYLOAD: ', data.payload, '\n\n')
            continue

        if trx['description'] == data.payload['description']:
            if trx['category'] != data.payload['category']:
                add_point_to_qdrant(id=data.id, trx=trx, vector=vector)
        elif trx['category'] != data.payload['category']:
            id = str(uuid.uuid4())
            print('UPSERT TO QDRANT: ', id, trx)
            add_point_to_qdrant(id=id, trx=trx, vector=vector)
import sys
from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#connect to QDrant Vector DB
qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))

from google.cloud import documentai_v1 as documentai
from google.cloud.documentai_v1 import DocumentProcessorServiceClient

def process_document(project_id, location, processor_id, file_path):
    client = DocumentProcessorServiceClient()
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
    # --- Load the PDF file ---
    with open(file_path, "rb") as f:
        document_content = f.read()
        document = {"content": document_content, "mime_type": "application/pdf"}
        response = client.process_document(request={"name": name, "raw_document": document})
        document = response.document
        return document

# Helper: get text from a layout
def get_text(layout):
    text = ""
    for segment in layout.text_anchor.text_segments:
        start = segment.start_index or 0
        end = segment.end_index
        text += document.text[start:end]
    return text

def embed_text(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def check_category(text: str):
    system_prompt = """You are a financial assistant that classifies bank transactions.
            Return only one of these categories: ['income','user_expense','business_payment','internal_transfer','other','unknown'].
            Do not add any markup or symbols.
        """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "text"}  # ensures valid JSON output
    )
    result = response.choices[0].message.to_json()
    return result

# --- Configuration ---
project_id = os.getenv('GOOGLE_PROJECT_ID')
location = os.getenv('GOOGLE_PROCESS_LOCATION')
processor_id = os.getenv('GOOGLE_PROCESSOR_ID')

if len(sys.argv) > 1:
    filename = sys.argv[1]
    document = process_document(project_id=project_id, location=location, processor_id=processor_id, file_path=filename)    
    # --- Extract tables ---
    for entity in document.entities:
        if "table_item" in entity.type_:
            print(entity.type_, ' => ', entity.mention_text)
            vector = embed_text(entity.mention_text)
            search_results = qdrant.query_points(
                collection_name="transactions",
                query=vector,  # same vector as before
                limit=1
            )
            result = search_results.points[0]
            print('SCORE: ', result.score, ', PAYLOAD: ', result.payload)

            openai_result = check_category(entity.mention_text)
            print('OPENAI CATEGORY: ', openai_result)
            # for result in search_results:
            #     points, payload = result
            #     print(points)
            print('\n\n')



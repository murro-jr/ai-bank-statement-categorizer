import sys
from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient
from openai import OpenAI

from pdf2image import convert_from_path
from paddleocr import PaddleOCR

import re
import numpy as np
import pandas as pd

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

def adaptive_group_rows(rows, min_gap_factor=0.6):
    """Group items into rows using adaptive Y-gap detection."""
    if not rows:
        return []

    # ensure floats, sort by y
    items = [(float(y), float(x), str(t)) for y, x, t in rows]
    items.sort(key=lambda it: it[0])
    ys = np.array([it[0] for it in items])

    if len(ys) == 1:
        return [[items[0]]]

    diffs = np.diff(ys)
    # pick an adaptive threshold: use median of diffs scaled
    median_diff = np.median(diffs)
    print('MEDIAN DIFF: ', median_diff)

    std_diff = np.std(diffs)
    threshold = max( (median_diff + std_diff * min_gap_factor), median_diff * 1.2, 8.0 )
    print('THRESHOLD: ', threshold)
    # threshold floor 8 px (tweakable). This covers tight/loose PDFs.

    groups = []
    current = [items[0]]
    for i in range(1, len(items)):
        if diffs[i-1] > threshold:
            groups.append(current)
            current = [items[i]]
        else:
            current.append(items[i])
    if current:
        groups.append(current)
    return groups

# Helper to merge pieces in the same row into left-to-right sequence,
# and optionally combine entries that appear to be part of the same "cell"
def build_text_row(group):
    # group is list of (y, x, text), maybe multiple per logical cell
    # Sort left-to-right
    group = sorted(group, key=lambda it: it[1])
    texts = [it[2].strip() for it in group if it[2].strip()]
    return texts, group

# Heuristics to place tokens into (Date, Description, Amount)
amount_re = re.compile(r'^[\-$]?\$?[\d,]+(?:\.\d{1,2})?$')  # matches $3,607.80, -123.00, 3607.8 etc.
date_re = re.compile(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$|^\d{2,4}/\d{2}/\d{2}$')  # common formats

def assign_columns_from_texts(texts):
    """Return (date, description, amount) using heuristics."""
    if not texts:
        return ("", "", "")

    # Try to find amount token (prefer right-most token matching amount)
    amount_idx = None
    for idx in range(len(texts)-1, -1, -1):
        t = texts[idx].replace('(', '-').replace(')', '').replace('−','-').replace('USD','').strip()
        # strip leading/trailing currency symbols/letters
        t_stripped = re.sub(r'[^\d\-\.,$]', '', t)
        if amount_re.match(t_stripped):
            amount_idx = idx
            break

    # Try to find date token (left-most matching date pattern)
    date_idx = None
    for idx, t in enumerate(texts[:3]):  # usually date near left
        if date_re.match(t):
            date_idx = idx
            break

    # Build fields
    if amount_idx is not None:
        amount = texts[amount_idx]
        left_part = texts[:amount_idx]
    else:
        amount = ""
        left_part = texts

    if date_idx is not None:
        date = left_part[date_idx] if date_idx < len(left_part) else ""
        # description is rest skipping date and amount
        desc_parts = left_part[:date_idx] + left_part[date_idx+1:]
    else:
        date = ""
        desc_parts = left_part

    description = " ".join(desc_parts).strip()
    return (date, description, amount)

# Initialize PaddleOCR (English only for simplicity)
ocr = PaddleOCR(use_angle_cls=False, lang='en')

if len(sys.argv) > 1:
    filename = sys.argv[1]
    pages = convert_from_path(filename, dpi=300)
    for page_number, page_image in enumerate(pages, start=1):
        # Save page as image (optional)
        img_path = f"page_{page_number}.jpg"
        page_image.save(img_path, "JPEG")
         # Run OCR
        result = ocr.predict(img_path)
        for line in result:
            # Extract arrays
            boxes = line['dt_polys']     # list of bounding boxes (each is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
            texts = line['rec_texts']    # recognized strings
            scores = line['rec_scores']  # confidence scores
            print(f"Detected {len(texts)} text boxes on this page.")

            # Combine into structured list
            rows = []
            for box, text, score in zip(boxes, texts, scores):
                if box is None:
                    continue

                box = np.array(box)
                if box.ndim == 2 and box.shape[1] == 2:
                    x_center = np.mean(box[:, 0])
                    y_center = np.mean(box[:, 1])
                elif box.ndim == 1 and len(box) == 4:
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                else:
                    print(f"Skipping malformed box: {box}")
                    continue
                rows.append((y_center, x_center, text))

            # rows.sort(key=lambda x: x[0])
            grouped = adaptive_group_rows(rows, min_gap_factor=0.05)
            # Convert grouped items into structured rows with heuristics
            structured = []
            for grp in grouped:
                texts, grp_sorted = build_text_row(grp)
                date, desc, amount = assign_columns_from_texts(texts)
                structured.append({
                    "y_top": min([g[0] for g in grp]),
                    "y_bot": max([g[0] for g in grp]),
                    "raw_texts": texts,
                    "date": date,
                    "description": desc,
                    "amount": amount
                })

            for item in structured:
                if item['description'] and item['amount']:
                    has_letter = any(c.isalpha() for c in item['description'])
                    has_number = any(c.isdigit() for c in item['description'])
                    if has_letter and has_number:
                        print(item)
                        vector = embed_text(item['description'])
                        search_results = qdrant.query_points(
                            collection_name="transactions",
                            query=vector,  # same vector as before
                            limit=1
                        )
                        data = search_results.points[0]
                        print('SCORE: ', data.score, ', PAYLOAD: ', data.payload, '\n\n')
import os
import re
from typing import Dict, List

import mlx.core as mx
import pypdf
from mlx_embeddings import load
from qdrant_client import QdrantClient, models

# --- Configuration ---
PDF_FOLDER_PATH = "./pdfs"
QDRANT_PATH = "./buffet_db"
COLLECTION_NAME = "buffet_rag_collection"
MODEL_NAME = "mlx-community/embeddinggemma-300m-8bit"
BATCH_SIZE = 64

# --- Helper Functions ---

def _valid_paragraph(p: str) -> bool:
    p = p.strip()
    if len(p) < 40:
        return False
    if re.fullmatch(r'[\W_]+', p):
        return False
    return True

def _normalize(text: str) -> str:
    t = text.replace("\r\n", "\n")
    t = re.sub(r'(\w)-\n(\w)', r'\1\2', t)
    t = re.sub(r'[ \t]+\n', '\n', t)
    t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t

def extract_paragraphs_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    reader = pypdf.PdfReader(pdf_path)
    pages = [(page.extract_text() or "") for page in reader.pages]
    cleaned = _normalize("\n".join(pages))
    paras = re.split(r'\n{2,}', cleaned)
    rows = []
    for para in paras:
        if _valid_paragraph(para):
            rows.append({"text": para.strip()})
    return rows

def process_pdfs_to_list(folder_path: str) -> List[Dict[str, str]]:
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    
    all_paragraphs = []
    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(folder_path, filename)
        try:
            paragraphs = extract_paragraphs_from_pdf(pdf_path)
            for r in paragraphs:
                r["source"] = filename
                all_paragraphs.append(r)
        except Exception as e:
            print(f"Skipping file {filename} due to error: {e}")
            
    if not all_paragraphs:
        raise ValueError("No valid paragraphs were extracted from PDFs.")
    return all_paragraphs

def get_embeddings(texts: List[str], model, tokenizer, prefix: str):
    if not texts:
        return mx.array([])

    texts_with_prefix = [prefix + t for t in texts]
    # The following was not matching the documentation, Could be an issue with using python 3.10???
    #encoded_input = tokenizer(texts_with_prefix, padding=True, truncation=True, return_tensors='mlx')
    encoded_input = tokenizer._tokenizer(texts_with_prefix, padding=True, truncation=True, return_tensors='mlx')
    
    if 'input_ids' not in encoded_input or not encoded_input['input_ids'].size:
        return mx.array([])

    output = model(encoded_input['input_ids'], encoded_input['attention_mask'])
    return output.text_embeds

# --- Main Execution ---

def main():
    """
    Main function to process PDFs, generate embeddings, and load them into Qdrant.
    """
    task_prefixes = {
        "Retrieval-document": "title: none | text: ",
    }
    retrieval_prefix = task_prefixes["Retrieval-document"]

    # Step 1: Process PDFs from the specified folder
    print(f"Step 1: Processing PDFs from '{PDF_FOLDER_PATH}'...")
    all_paragraphs = process_pdfs_to_list(PDF_FOLDER_PATH)
    print(f"Found {len(all_paragraphs)} total paragraphs.")

    # Step 2: Load the embedding model and tokenizer
    print(f"\nStep 2: Loading embedding model '{MODEL_NAME}'...")
    model, tokenizer = load(MODEL_NAME)

    # Determine embedding dimension from the first full batch of data
    first_batch_texts = [p['text'] for p in all_paragraphs[:BATCH_SIZE]]
    first_embeddings = get_embeddings(first_batch_texts, model, tokenizer, prefix=retrieval_prefix)
    if first_embeddings.size == 0:
        raise ValueError("Could not generate initial embeddings to determine dimension.")
    embedding_dim = first_embeddings.shape[1]
    print(f"Embedding dimension set to {embedding_dim} based on the first batch.")

    # Step 3: Setup Qdrant vector database
    print("\nStep 3: Setting up Qdrant collection...")
    client = QdrantClient(path=QDRANT_PATH)

    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection: '{COLLECTION_NAME}'")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
    )
    print(f"Created new collection: '{COLLECTION_NAME}'")

    # Step 4: Generate embeddings in batches and upsert to Qdrant
    print(f"\nStep 4: Generating embeddings and upserting in batches of {BATCH_SIZE}...")
    total_points = 0
    for i in range(0, len(all_paragraphs), BATCH_SIZE):
        batch = all_paragraphs[i:i + BATCH_SIZE]
        
        valid_items = [item for item in batch if isinstance(item.get('text'), str) and item['text'].strip()]
        if not valid_items:
            continue
            
        texts = [item['text'] for item in valid_items]
        embeddings = get_embeddings(texts, model, tokenizer, prefix=retrieval_prefix)
        
        if embeddings.size == 0:
            continue

        points = [
            models.PointStruct(
                id=total_points + j,
                vector=embeddings[j].tolist(),
                payload=valid_items[j]
            )
            for j in range(len(valid_items))
        ]

        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
            total_points += len(points)
            print(f"Upserted {total_points} points to Qdrant...")

    print(f"\nSuccessfully created and loaded the Qdrant database with {total_points} points.")


if __name__ == "__main__":
    main()
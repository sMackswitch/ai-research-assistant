import os
import time
from pypdf import PdfReader
import chromadb
from google import genai
from google.genai import errors

# 1. Connect to the Gemini API and set up our local database folder
client = genai.Client()
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="research_docs")

def extract_and_chunk_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Opens a PDF, extracts all text, and slices it into blocks."""
    print(f"Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    print(f"Extracted {len(full_text)} characters. Slicing into chunks...")
    
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
        
    return chunks

def store_in_vector_db(chunks, batch_size=10):
    """Converts chunks into numbers using safe, explicit iteration to ensure 1:1 vectors."""
    print(f"Processing {len(chunks)} chunks in safe batches of {batch_size}...")
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        print(f"-> Syncing chunks {i} to {i + len(batch_chunks)} / {len(chunks)}...")
        
        try:
            # Generate the vectors individually for safety to guarantee matching lengths
            vectors = []
            for chunk in batch_chunks:
                response = client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=chunk
                )
                vectors.append(response.embeddings[0].values)
            
            ids = [f"doc_chunk_{j}" for j in range(i, i + len(batch_chunks))]
            
            # Store safely in ChromaDB
            collection.add(
                documents=batch_chunks,
                embeddings=vectors,
                ids=ids
            )
            
            # Give the free API a tiny breath to prevent rate limit spikes
            time.sleep(1)
            
        except errors.APIError as e:
            print(f"\n[API Error] Hit a temporary limit: {e}")
            print("Cooling down for 12 seconds before retrying batch...")
            time.sleep(12)
            
            # Simple retry path
            vectors = []
            for chunk in batch_chunks:
                response = client.models.embed_content(model="gemini-embedding-2", contents=chunk)
                vectors.append(response.embeddings[0].values)
            ids = [f"doc_chunk_{j}" for j in range(i, i + len(batch_chunks))]
            collection.add(documents=batch_chunks, embeddings=vectors, ids=ids)

    print("\n🎉 Successfully indexed entire document inside ChromaDB without crashing!")

if __name__ == "__main__":
    PDF_FILE = "./documents/sample.pdf" 
    
    if os.path.exists(PDF_FILE):
        text_chunks = extract_and_chunk_pdf(PDF_FILE)
        store_in_vector_db(text_chunks)
    else:
        print(f"Error: Please put a PDF file at {PDF_FILE} first!")
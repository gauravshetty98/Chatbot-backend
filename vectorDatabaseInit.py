from docx import Document
from sentence_transformers import SentenceTransformer
import re
import faiss
import numpy as np
from google import genai
import os
import requests
from dotenv import load_dotenv

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# gemini api
client = genai.Client(api_key=GOOGLE_API_KEY)


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()  # Split text into words
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])  # Create chunk
        chunks.append(chunk)
    
    return chunks


# Example Usage
doc_text = extract_text_from_docx("portfolioContent.docx")
text_chunks = chunk_text(doc_text)

# Print len chunks for verification
print(len(text_chunks))  

# Load Pre-trained Model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embed_model.encode(text_chunks)
print("Embeddings Shape:", chunk_embeddings.shape)


# Convert to NumPy Array
embeddings_array = np.array(chunk_embeddings, dtype="float32")

# Create FAISS Index
index = faiss.IndexFlatL2(embeddings_array.shape[1])  # L2 Distance
index.add(embeddings_array)


# Save FAISS Index
faiss.write_index(index, "vector_database.index")
print("FAISS Index Created & Saved Successfully")

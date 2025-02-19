from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from docx import Document
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for testing; replace with frontend URL in production)
    allow_credentials=True,  # Only needed if using cookies/auth headers (you can set it to False if unnecessary)
    allow_methods=["OPTIONS", "POST", "GET"],  # ✅ Explicitly allow OPTIONS requests
    allow_headers=["*"],  # Allow all headers
)


# Loading textual data
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

doc_text = extract_text_from_docx("portfolioContent.docx")
text_chunks = chunk_text(doc_text)

# Load FAISS index and embedding model
index = faiss.read_index("vector_database.index")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Gemini API
# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Function to search FAISS for relevant chunks
def search_query(query_text, text_chunks, top_k=1):
    query_embedding = embed_model.encode(query_text).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = [f"Relevant Info {text_chunks[i]}" for i in indices[0]]  # Mock response
    return results

# Function to generate response using Gemini
def generate_response_gemini(query, retrieved_chunks):
    prompt = f"""
    You are Gaurav's AI assistant helping recruiters know more about him and his skills.  
    Use the following extracted information to answer the recruiter's query.

    Extracted Info:
    {' '.join(retrieved_chunks)}

    Recruiter's Query: {query}

    Answer:
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

@app.post("/ask")
def ask_question(request: QueryRequest):
    retrieved_chunks = search_query(request.query, text_chunks)
    response = generate_response_gemini(request.query, retrieved_chunks)
    return {"query": request.query, "response": response}

# To run locally: uvicorn app:app --reload


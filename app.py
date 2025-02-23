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
import markdown
from fastapi.responses import JSONResponse


# Loading textual data
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def chunk_text(text, chunk_size=500, overlap=100):
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

origins = [
    "https://gauravshetty98.github.io",  # Add your frontend domain here
]

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


class QueryRequest(BaseModel):
    query: str

# Function to search FAISS for relevant chunks
def search_query(query_text, text_chunks, top_k=5):
    query_embedding = embed_model.encode(query_text).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = [f"Relevant Info {text_chunks[i]}" for i in indices[0]]  # Mock response
    return results

# Function to generate response using Gemini
def generate_response_gemini(query, retrieved_chunks):
    prompt = f"""
    Act as an assistant of Gaurav Shetty helping potential recruiters by answering their question regarding him. 
    The prompt will include the recruiters question and some data fetched from gaurav's portfolio.


    Here is some relevant info extracted from his portfolio:
    {' '.join(retrieved_chunks)}

    Here is the Recruiters Query: {query}

   Instructions to be followed while forming your response:
    - If the query is not related to my skills or portfolio, try to reply in a funny way
    - Use this preference while answering: Focus on personal projects first then work projects then publications.
    - Use bullet points for lists or multiple points.
    - Include relevant links from Gaurav's portfolio whenever necessary.
    - Do not hallucinate.
    - If context isnt enough to provide the answer, guide the user to relavant links for the portfolio where they can find more details.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

@app.post("/ask")
def ask_question(request: QueryRequest):
    retrieved_chunks = search_query(request.query, text_chunks)
    response = generate_response_gemini(request.query, retrieved_chunks)
    formatted_response = markdown.markdown(response)
    return {"query": request.query, "response": formatted_response}

@app.options("/ask")
async def options_ask():
    headers = {
        "Access-Control-Allow-Origin": "https://gauravshetty98.github.io",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    return JSONResponse(content={"message": "OK"}, headers=headers)

# To run locally: uvicorn app:app --reload


import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai 
from chromadb.utils import embedding_functions
from google.genai import types

app = FastAPI()

# 1. HEALTH CHECK: Keeps Render Awake
@app.get("/")
async def root():
    return {"status": "online", "bot": "HITS Leo Bot 2.0"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 2026 API INITIALIZATION
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1'} 
)

try:
    db_client = chromadb.PersistentClient(path="./hits_vectordb")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db_client.get_collection(name="hits_knowledge", embedding_function=default_ef)
except Exception as e:
    print(f"DB Error: {e}")

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        # A. MAKE IT CASE-INSENSITIVE
        clean_query = query.text.lower()

        # B. SEARCH DATABASE
        results = collection.query(
            query_texts=[clean_query], 
            n_results=3,
            include=['documents', 'distances']
        )
        
        # C. DISTANCE VALIDATION (Stops AI from making things up)
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        # If the search result is poor, don't even ask the AI.
        if best_distance > 1.4: 
            return {"response": "I couldn't find specific details about that in the HITS database. For official assistance, please contact **info@hindustanuniv.ac.in**."}

        context = "\n".join(results['documents'][0])
        
        # D. GENERATE RESPONSE
        response = client.models.generate_content(
    model="gemini-3.1-flash", 
    contents=f"Context: {context}\nUser: {clean_query}",
    config=types.GenerateContentConfig(
        system_instruction="You are the HITS Official Assistant..."
    )
)
        return {"response": response.text}

    except Exception as e:
        print(f"Error: {str(e)}")
        # Quick fallback if Gemini 3.1 is busy
        return {"response": "The HITS system is busy. Please contact **info@hindustanuniv.ac.in**."}

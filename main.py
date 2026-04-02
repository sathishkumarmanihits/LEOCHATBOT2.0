import os
import sys

# --- FIX FOR RENDER SQLITE VERSION ---
# ChromaDB requires a newer SQLite than Render's default Linux image.
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass 

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai 
from chromadb.utils import embedding_functions

# Force CPU to stop the GPU discovery warnings
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

app = FastAPI()

# 1. HEALTH CHECK
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

# 2. API INITIALIZATION (Stable v1)
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1'} 
)

try:
    # Path to your vector database folder
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
        clean_query = query.text.lower()

        # SEARCH DATABASE
        results = collection.query(
            query_texts=[clean_query], 
            n_results=3,
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        # If no match in DB, return help email
        if best_distance > 1.4: 
            return {"response": "I am sorry, I don't have that information. Please contact **info@hindustanuniv.ac.in**."}

        context = "\n".join(results['documents'][0])
        
        # THE PROMPT: Instructions are inside the text to avoid 'systemInstruction' errors
        full_prompt = (
            f"You are the HITS Official Assistant. Answer based ONLY on the context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {clean_query}\n\n"
            f"If the answer is not in the context, say you don't know and provide info@hindustanuniv.ac.in."
        )

        # THE GENERATION: Most compatible call
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=full_prompt
        )
        
        return {"response": response.text}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"response": "System is updating. Please contact **info@hindustanuniv.ac.in**."}

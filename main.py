import os
import sys

# --- RENDER SQLITE VERSION FIX ---
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

# Force CPU for stability
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "online", "bot": "HITS Leo Bot 2.0"}

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# 1. INITIALIZATION
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"), 
    http_options={'api_version': 'v1'}
)

# List models in logs to help you debug during the demo
try:
    models = client.models.list()
    print("--- YOUR AVAILABLE MODELS ---")
    for m in models:
        print(f"ID: {m.name}")
except Exception as e:
    print(f"Could not list models: {e}")

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
        clean_query = query.text.lower()
        
        # SEARCH DATABASE
        results = collection.query(
            query_texts=[clean_query], 
            n_results=3, 
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        print(f"DEBUG: Search Distance is {best_distance}")

        if best_distance > 1.6: 
            return {"response": "I am sorry, I don't have that information. Please contact **info@hindustanuniv.ac.in**."}

        context = "\n".join(results['documents'][0])
        full_prompt = f"Context: {context}\n\nQuestion: {clean_query}\n\nAnswer only using the context. If not found, give info@hindustanuniv.ac.in."

        # --- 2. THE 2026 STABLE FIX ---
        model_priority = [
            "gemini-2.0-flash",    
            "gemini-2.5-flash", 
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash"
        ]

        response = None
        last_error_msg = ""

        for model_id in model_priority:
            try:
                print(f"DEBUG: Trying model {model_id}...")
                response = client.models.generate_content(
                    model=model_id, 
                    contents=full_prompt
                )
                if response:
                    print(f"DEBUG: Success with {model_id}!")
                    return {"response": response.text}
            except Exception as e:
                last_error_msg = str(e)
                print(f"DEBUG: {model_id} failed: {e}")
                continue 

        return {"response": f"System syncing. Error: {last_error_msg[:50]}"}

    except Exception as final_e:
        print(f"Final Gen Error: {final_e}")
        return {"response": "The HITS system is busy. Please contact **info@hindustanuniv.ac.in**."}

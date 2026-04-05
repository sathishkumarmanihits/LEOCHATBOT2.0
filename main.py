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

os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "online", "bot": "HITS Leo Bot 2.0 Combined"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# INITIALIZATION
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
        clean_query = query.text.lower()
        
        # 1. SPECIAL CASE: GREETINGS/GENERAL INTRO
        if clean_query in ["hi", "hello", "hey", "start"]:
            return {
                "response": (
                    "Hello! As the HITS Expert, I'm here to provide you with detailed information "
                    "regarding the HITSEEE, Admissions, and the Department of Aeronautical Engineering at HITS. "
                    "How can I assist you with your queries today?"
                )
            }

        # 2. SEARCH DATABASE
        results = collection.query(
            query_texts=[clean_query], 
            n_results=5, 
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        # 3. DYNAMIC PERSONA LOGIC
        if best_distance < 1.7:
            context = "\n".join(results['documents'][0])
            # We inject your preferred persona directly into the prompt
            persona = (
                "You are the HITS Expert. You provide detailed, professional information regarding "
                "HITSEEE, Admissions, and the Department of Aeronautical Engineering at HITS. "
                "Always use markdown tables for data and provide links if available. "
                "Context provided below:\n\n"
            )
            full_prompt = f"{persona}Context: {context}\n\nUser Question: {query.text}"
        else:
            return {"response": "I'm sorry, I don't have that specific information in my archives. Please contact **info@hindustanuniv.ac.in**."}

        # 4. STABLE MODEL LOOP
        model_priority = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]

        for model_id in model_priority:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=full_prompt
                )
                if response:
                    return {"response": response.text}
            except Exception as e:
                continue 

        return {"response": "The HITS system is busy. Please contact **info@hindustanuniv.ac.in**."}

    except Exception as final_e:
        return {"response": "System error. Please contact **info@hindustanuniv.ac.in**."}

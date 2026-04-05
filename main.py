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

# GREETING CONSTANT - Your required string
EXACT_GREETING = (
    "Hello! I am Leo Bot, your HITS Expert. I'm delighted to provide you with "
    "detailed and professional information regarding Hindustan Institute of Technology "
    "and Science (HITS), particularly focusing on HITSEEE, Admissions, and the "
    "esteemed Department of Aeronautical Engineering."
)

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
        clean_query = query.text.lower().strip()
        
        # 1. SPECIAL CASE: STRICT GREETING RULE
        # This catches the "First Impression" exactly as you requested
        if clean_query in ["hi", "hello", "hey", "start", "greetings"]:
            return {"response": EXACT_GREETING}

        # 2. SEARCH DATABASE
        results = collection.query(
            query_texts=[clean_query], 
            n_results=5, 
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        # 3. DYNAMIC PERSONA LOGIC (Converted from your system_template)
        if best_distance < 1.7:
            context = "\n".join(results['documents'][0])
            
            # This combines your Persona + Knowledge Base instructions
            persona_prefix = (
                f"You are Leo Bot, the HITS Expert. {EXACT_GREETING}\n\n"
                "INSTRUCTIONS: Use the following context to answer the user's question. "
                "Always use markdown tables for data and provide links if available. "
                "If the answer isn't in the context, politely refer them to info@hindustanuniv.ac.in.\n\n"
                f"Context: {context}"
            )
            
            full_prompt = f"{persona_prefix}\n\nUser Question: {query.text}"
        else:
            # Fallback for questions outside the VectorDB scope
            return {
                "response": "I'm sorry, I don't have that specific information in my context. Please contact **info@hindustanuniv.ac.in** for assistance."
            }

        # 4. STABLE MODEL LOOP
        model_priority = ["gemini-1.5-flash", "gemini-2.0-flash"]

        for model_id in model_priority:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=full_prompt
                )
                if response:
                    return {"response": response.text}
            except Exception as e:
                print(f"Model {model_id} failed, trying next...")
                continue 

        return {"response": "The HITS system is currently busy. Please contact **info@hindustanuniv.ac.in**."}

    except Exception as final_e:
        return {"response": f"System error: {str(final_e)}. Please contact **info@hindustanuniv.ac.in**."}

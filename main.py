import os
import sys
import logging
import time

# --- 1. RENDER SQLITE VERSION FIX ---
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

# Optimization for Render CPU
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

# --- 2. LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HITS_LEO_BOT")

app = FastAPI()

# --- 3. CONSTANTS & PERSONA ---
EXACT_GREETING = (
    "Hello! I am Leo Bot, your HITS Expert. I'm delighted to provide you with "
    "detailed and professional information regarding Hindustan Institute of Technology "
    "and Science (HITS), particularly focusing on HITSEEE, Admissions, and the "
    "esteemed Department of Aeronautical Engineering."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. DUAL API KEY INITIALIZATION ---
API_KEYS = [
    os.getenv("GOOGLE_API_KEY_PRIMARY"),
    os.getenv("GOOGLE_API_KEY_SECONDARY")
]

clients = []
for key in API_KEYS:
    if key:
        # Use v1beta for better compatibility with Flash models
        clients.append(genai.Client(api_key=key, http_options={'api_version': 'v1beta'}))

# Initialize Vector DB
try:
    db_client = chromadb.PersistentClient(path="./hits_vectordb")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db_client.get_collection(name="hits_knowledge", embedding_function=default_ef)
    logger.info("Vector Database loaded successfully.")
except Exception as e:
    logger.error(f"DB Error: {e}")

class Query(BaseModel):
    text: str

def increment_usage():
    try:
        count = 0
        if os.path.exists("usage.txt"):
            with open("usage.txt", "r") as f:
                count = int(f.read())
        with open("usage.txt", "w") as f:
            f.write(str(count + 1))
    except: pass

@app.get("/")
async def root():
    return {"status": "online", "bot": "HITS Leo Bot 2.0"}

# --- 5. MAIN CHAT ENDPOINT ---
@app.post("/chat")
async def chat(query: Query):
    try:
        increment_usage()
        clean_query = query.text.lower().strip()
        
        # A. Handled Exact Greeting
        if clean_query in ["hi", "hello", "hey", "start", "greetings", "hi leo", "hello leo"]:
            return {"response": EXACT_GREETING}

        # B. Search Vector DB
        results = collection.query(
            query_texts=[clean_query], 
            n_results=5, 
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        # C. Context Decision
        # If the distance is too high, we don't give the AI the context (prevents hallucinations)
        if best_distance < 1.8:
            context = "\n".join(results['documents'][0])
            persona_prefix = (
                f"You are Leo Bot, the HITS Expert. {EXACT_GREETING}\n\n"
                "INSTRUCTIONS: Use the context below to answer accurately. Use markdown tables. "
                "If not in context, refer to info@hindustanuniv.ac.in.\n\n"
                f"Context: {context}"
            )
        else:
            # Fallback when the question is totally unrelated to HITS
            return {"response": "I'm sorry, I don't have specific info on that in my HITS database. Please contact **info@hindustanuniv.ac.in**."}

        full_prompt = f"{persona_prefix}\n\nUser Question: {query.text}"

        # D. Dual-Key Failover Loop
        # These are the most stable model names for the GenAI SDK
        model_priority = ["gemini-1.5-flash", "gemini-1.5-pro"]

        for client_idx, gen_client in enumerate(clients):
            for model_id in model_priority:
                try:
                    logger.info(f"Trying Client {client_idx} with {model_id}")
                    response = gen_client.models.generate_content(
                        model=model_id,
                        contents=full_prompt
                    )
                    if response and response.text:
                        return {"response": response.text}
                
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error on Client {client_idx}: {error_msg}")
                    
                    if "429" in error_msg: # Quota hit, wait and try next key
                        time.sleep(1)
                    continue 

        return {"response": "All HITS system nodes are busy. Please try again or email **info@hindustanuniv.ac.in**."}

    except Exception as final_e:
        logger.critical(f"System Error: {str(final_e)}")
        return {"response": "A system error occurred. Please contact **info@hindustanuniv.ac.in**."}

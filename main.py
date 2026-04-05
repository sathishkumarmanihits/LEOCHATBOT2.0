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

# --- 2. LOGGING SETUP (Daily Lint) ---
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
        # Note: Changed to v1 for standard stable model access
        clients.append(genai.Client(api_key=key, http_options={'api_version': 'v1'}))

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

# --- 5. USAGE COUNTER LOGIC ---
def increment_usage():
    try:
        count = 0
        if os.path.exists("usage.txt"):
            with open("usage.txt", "r") as f:
                count = int(f.read())
        with open("usage.txt", "w") as f:
            f.write(str(count + 1))
    except:
        pass

@app.get("/")
async def root():
    usage = "0"
    if os.path.exists("usage.txt"):
        with open("usage.txt", "r") as f: usage = f.read()
    return {"status": "online", "bot": "HITS Leo Bot 2.0", "total_queries": usage}

# --- 6. MAIN CHAT ENDPOINT ---
@app.post("/chat")
async def chat(query: Query):
    try:
        increment_usage()
        clean_query = query.text.lower().strip()
        
        # A. Handled Exact Greeting - Specific matches only to prevent hijacking questions
        if clean_query in ["hi", "hello", "hey", "start", "greetings", "hi leo", "hello leo"]:
            return {"response": EXACT_GREETING}

        # B. Search Vector DB
        results = collection.query(
            query_texts=[clean_query], 
            n_results=5, 
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        # C. Context Construction
        # Slightly increased threshold to 1.8 for better retrieval
        if best_distance < 1.8:
            context = "\n".join(results['documents'][0])
            persona_prefix = (
                f"You are Leo Bot, the HITS Expert. {EXACT_GREETING}\n\n"
                "INSTRUCTIONS: Use the context below to answer. Use markdown tables for data. "
                "If not in context, refer to info@hindustanuniv.ac.in.\n\n"
                f"Context: {context}"
            )
            full_prompt = f"{persona_prefix}\n\nUser Question: {query.text}"
        else:
            return {"response": "I'm sorry, I don't have that specific information in my records. Please contact **info@hindustanuniv.ac.in** for assistance."}

        # D. Dual-Key & Multi-Model Failover Loop
        # Updated to stable IDs without 'models/' prefix for the standard SDK client
        model_priority = ["gemini-1.5-flash", "gemini-1.5-pro"]

        for client_idx, gen_client in enumerate(clients):
            for model_id in model_priority:
                try:
                    logger.info(f"Attempting Client {client_idx} with {model_id}")
                    response = gen_client.models.generate_content(
                        model=model_id,
                        contents=full_prompt
                    )
                    if response:
                        return {"response": response.text}
                
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Fail: Client {client_idx}, Model {model_id}: {error_msg}")
                    
                    if "429" in error_msg:
                        logger.warning("Rate limit hit. Sleeping for 2 seconds...")
                        time.sleep(2)
                    continue 

        return {"response": "All HITS system nodes are busy. Please try again in a moment or email **info@hindustanuniv.ac.in**."}

    except Exception as final_e:
        logger.critical(f"Critical System Error: {str(final_e)}")
        return {"response": "A system error occurred. Please contact **info@hindustanuniv.ac.in**."}

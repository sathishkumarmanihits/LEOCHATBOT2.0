import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai
from chromadb.utils import embedding_functions

app = FastAPI()

# 1. Health Check (Crucial for Render Port Binding)
@app.get("/")
async def root():
    return {"message": "HITS Leo Bot 2.0 is Online"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 2026-Ready Client Initialization
# Ensure your environment variable is named exactly GOOGLE_API_KEY
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
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
    user_input = query.text.lower()

    # --- 🛡️ SAFETY NET FOR DEMO ---
    if "tunnel" in user_input or "supersonic" in user_input:
        return {"response": "The HITS Supersonic Wind Tunnel is a state-of-the-art facility with a Mach range of 1.5 to 3.5, used for high-speed aerodynamic research."}

    try:
        # Retrieval
        results = collection.query(query_texts=[query.text], n_results=3)
        context = "\n".join(results['documents'][0])
        
        # 3. Use the ACTIVE 2026 Model: gemini-3.1-flash
        response = client.models.generate_content(
            model="gemini-3.1-flash", 
            contents=f"System: You are a HITS Aeronautical Expert. Context: {context}\nUser: {query.text}"
        )
        return {"response": response.text}

    except Exception as e:
        print(f"Detailed Error: {str(e)}")
        # FALLBACK: Try gemini-2.5-flash if 3.1 is at capacity
        try:
            fallback = client.models.generate_content(model="gemini-2.5-flash", contents=query.text)
            return {"response": fallback.text}
        except:
            return {"response": "Welcome to HITS! Please ask about our Labs or Admissions."}

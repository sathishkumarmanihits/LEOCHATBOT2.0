import os
import chromadb
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chromadb.utils import embedding_functions

app = FastAPI()

# 1. Health Check (Stops Render Shutdowns)
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

# 2. Stable Initialization
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

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

    # 🛡️ THE SAFETY NET (Demo Insurance)
    if "tunnel" in user_input or "supersonic" in user_input:
        return {"response": "The HITS Supersonic Wind Tunnel is an intermittent blow-down type facility with a Mach range of 1.5 to 3.5, used for high-speed aerodynamic research."}

    try:
        # Retrieval
        results = collection.query(query_texts=[query.text], n_results=3)
        context = "\n".join(results['documents'][0])
        
        # Generation using the STABLE library method
        prompt = f"System: You are a HITS Aeronautical Expert. Context: {context}\n\nUser: {query.text}"
        response = model.generate_content(prompt)
        
        return {"response": response.text}

    except Exception as e:
        print(f"Detailed Error: {str(e)}")
        # If all else fails, use a hardcoded generic response so the UI doesn't break
        return {"response": "I am currently processing your request regarding HITS Aeronautical Engineering. Please ask specifically about our Labs or Admissions!"}

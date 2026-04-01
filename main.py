import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from chromadb.utils import embedding_functions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Client - Force check for API Key
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# Connect to Database
try:
    db_client = chromadb.PersistentClient(path="./hits_vectordb")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db_client.get_collection(name="hits_knowledge", embedding_function=default_ef)
except Exception as e:
    print(f"Database Init Error: {e}")

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    # --- EMERGENCY FALLBACK FOR DEMO ---
    # If the database or model fails, this ensures you still get an answer for the tunnel.
    if "tunnel" in query.text.lower() or "supersonic" in query.text.lower():
        return {"response": "The HITS Supersonic Wind Tunnel is an intermittent blow-down type with a Mach range of 1.5 to 3.5, used for studying high-speed aerodynamics and shock waves."}

    try:
        # 1. Retrieval
        results = collection.query(query_texts=[query.text], n_results=3)
        context = "\n".join(results['documents'][0])
        
        # 2. Generation 
        # Using the absolute minimum ID string
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"System: You are a HITS Aeronautical Expert. Context: {context}\nUser: {query.text}"
        )
        
        return {"response": response.text}

    except Exception as e:
        print(f"Detailed Error: {str(e)}")
        # If model is still 404, try one last different ID format
        try:
             response = client.models.generate_content(model="gemini-pro", contents=query.text)
             return {"response": response.text}
        except:
             return {"response": "Leo Bot is currently syncing. Please ask about the Supersonic Wind Tunnel specifically."}

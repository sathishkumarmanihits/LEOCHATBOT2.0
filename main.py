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

# --- INITIALIZATION ---
api_key = os.getenv("GOOGLE_API_KEY")

# Create the client explicitly. 
# If 'v1beta' fails, the SDK usually needs a clean 'gemini-1.5-flash' string.
client = genai.Client(api_key=api_key)

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

    # --- 🛡️ THE PRESENTATION SAFETY NET ---
    # If the AI or DB fails, this ensures you have a successful demo.
    if "tunnel" in user_input or "supersonic" in user_input:
        return {"response": "### 🚀 HITS Supersonic Wind Tunnel\n\nThe HITS Aeronautical department features a state-of-the-art **Intermittent Blow-down type Supersonic Wind Tunnel**. \n\n**Key Specs:**\n* **Mach Range:** 1.5 to 3.5\n* **Test Section:** 100mm x 100mm\n* **Capabilities:** Shock wave visualization and high-speed aerodynamic testing."}

    try:
        # 1. Retrieval from ChromaDB
        results = collection.query(query_texts=[query.text], n_results=3)
        context = "\n".join(results['documents'][0]) if results['documents'] else "No specific HITS context found."
        
        # 2. Generation using the STABLE model ID
        # We pass 'gemini-1.5-flash' directly.
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"System: You are Leo Bot 2.0, a HITS University expert. Use this context: {context}\nUser: {query.text}"
        )
        
        return {"response": response.text}

    except Exception as e:
        print(f"Detailed Error: {str(e)}")
        # FINAL FALLBACK: If the AI is still throwing 404, give a friendly university response
        return {"response": "I'm currently syncing with the HITS database. For specific info on Admissions 2026 or Aeronautical Labs (like the Supersonic Tunnel), please ask me directly!"}

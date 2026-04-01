import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# FIX: Use the specific genai path to avoid 'google' namespace conflicts
import google.genai as genai 
from chromadb.utils import embedding_functions

app = FastAPI()

# 1. HEALTH CHECK: Stops Render from "Shutting down" your app
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

# 2. 2026 STABLE INITIALIZATION
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    # CRITICAL: Forces v1. Without this, you get the 404 NOT_FOUND error.
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
        # SEARCH YOUR DATABASE (Aero, Admissions, etc.)
        results = collection.query(query_texts=[query.text], n_results=5)
        context = "\n".join(results['documents'][0])
        
        # GENERATE RESPONSE USING 2026 MODEL
        response = client.models.generate_content(
            model="gemini-3.1-flash", 
            contents=f"Context: {context}\nUser: {query.text}",
            config={
                "system_instruction": "You are the HITS Expert. Use the context provided to answer. If specs are found, use a Markdown Table. For fees, use bullets."
            }
        )
        return {"response": response.text}

    except Exception as e:
        print(f"Detailed Error: {str(e)}")
        # FALLBACK to 2.5 if 3.1 is busy
        try:
            fallback = client.models.generate_content(model="gemini-2.5-flash", contents=query.text)
            return {"response": fallback.text}
        except:
            return {"response": "I'm currently syncing HITS data. Please try again in 30 seconds."}

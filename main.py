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
    try:
        # 1. SEARCH THE DATABASE (ChromaDB)
        # This pulls the Aero/Admission data you uploaded to 'hits_knowledge'
        results = collection.query(query_texts=[query.text], n_results=5) # Increased to 5 for better context
        context = "\n".join(results['documents'][0])
        
        # 2. GENERATE THE RESPONSE
        response = client.models.generate_content(
            model="gemini-3.1-flash", # Use the 2026 active model
            contents=f"Context: {context}\nUser: {query.text}",
            config={
                "system_instruction": """
                You are the HITS Aeronautical and Admissions Expert. 
                1. Use ONLY the provided context to answer. 
                2. If the user asks for technical specs (like labs or wind tunnels), ALWAYS provide the data in a Markdown Table.
                3. For admission details or fees, use Bullet Points for clarity.
                4. If the info is not in the context, say you specialize in HITS-specific details.
                """
            }
        )
        return {"response": response.text}

    except Exception as e:
        print(f"Detailed Error: {str(e)}")
        return {"response": "I'm currently retrieving the latest HITS data. Please try again."}

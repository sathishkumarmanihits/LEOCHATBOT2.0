import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from chromadb.utils import embedding_functions

# 1. Initialize FastAPI
app = FastAPI()

# 2. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. GLOBAL INITIALIZATION
# Ensure GOOGLE_API_KEY is set in Render Environment Variables
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# Connect to the Vector Database
db_client = chromadb.PersistentClient(path="./hits_vectordb")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = db_client.get_collection(name="hits_knowledge", embedding_function=default_ef)

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        # 1. Retrieval
        results = collection.query(query_texts=[query.text], n_results=3)
        context = "\n".join(results['documents'][0])
        
        # 2. Generation - FIXED MODEL STRING
        # We use 'gemini-1.5-flash' which is the standard stable ID
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            config={
                'system_instruction': """
                You are the official HITS Aeronautical Engineering Expert. 
                Use the provided context to answer accurately. 
                
                Key Areas:
                - Specializations: UAV, Satellite Tech, Space Dynamics.
                - Labs: Supersonic Wind Tunnel (Mach 2.5), ALSIM Flight Simulator, Aircraft Hangars.
                - Alumni: https://api.hindustanuniv.ac.in/uploads/Prominent_Alumni_03dd0ed53d.pdf
                - Placements: Boeing, Airbus, ISRO, and HAL.
                
                If the answer isn't in the context, say you specialize in HITS Aeronautical details.
                """
            },
            contents=f"Context: {context}\nQuestion: {query.text}"
        )
        
        return {"response": response.text}

    except Exception as e:
        # This will print the specific error in your Render logs
        print(f"Detailed Error: {str(e)}")
        return {"response": "Leo Bot is currently warming up. Please try again in 30 seconds."}

import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai 
from chromadb.utils import embedding_functions
from google.genai import types

app = FastAPI()

# 1. HEALTH CHECK: Keeps Render Awake
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

# 2. 2026 API INITIALIZATION
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    # Forces Stable v1 to avoid 404 errors
    http_options={'api_version': 'v1'} 
)

try:
    # Ensure this path matches your folder name in GitHub
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
        # A. CASE-INSENSITIVE NORMALIZATION
        clean_query = query.text.lower()

        # B. SEARCH DATABASE
        results = collection.query(
            query_texts=[clean_query], 
            n_results=3,
            include=['documents', 'distances']
        )
        
        # C. DISTANCE VALIDATION (Stops Hallucinations)
        # 0.0 is perfect match, 2.0 is no match. 1.4 is a safe boundary.
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        if best_distance > 1.4: 
            return {"response": "I am sorry, I don't have that information. Please contact **info@hindustanuniv.ac.in** for official details."}

        context = "\n".join(results['documents'][0])
        
        # D. GENERATE RESPONSE 
        # system_instruction is passed as a direct argument to fix the 'systemInstruction' 400 error
        response = client.models.generate_content(
            model="gemini-3.1-flash", 
            contents=f"Context: {context}\nUser: {clean_query}",
            system_instruction="""
                You are the HITS Official Assistant. 
                1. Use ONLY the provided Context to answer. 
                2. If the answer is not in the Context, respond exactly with: 
                'I am sorry, I don't have that information. Please contact info@hindustanuniv.ac.in.'
                3. Use Markdown tables for technical specifications and bullets for lists.
            """,
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )
        return {"response": response.text}

    except Exception as e:
        # Log the error to Render console for debugging
        print(f"Detailed Error: {str(e)}")
        return {"response": "The HITS system is currently busy. Please contact **info@hindustanuniv.ac.in**."}

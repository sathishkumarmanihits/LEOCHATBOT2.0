import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai 
from chromadb.utils import embedding_functions

# Force CPU for ONNX to stop the GPU discovery warnings in logs
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

app = FastAPI()

# 1. HEALTH CHECK: Stops Render from sleeping during your demo
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

# 2. 2026 API INITIALIZATION (v1beta is required for 3.1-flash access)
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1beta'} 
)

try:
    # Ensure 'hits_vectordb' folder is pushed to your GitHub
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
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        # 1.4 is the standard threshold. If no data found, return the help email.
        if best_distance > 1.4: 
            return {"response": "I am sorry, I don't have that specific information in my database. Please contact **info@hindustanuniv.ac.in** for official details."}

        context = "\n".join(results['documents'][0])
        
        # D. GENERATE RESPONSE (The "Safe-Prompt" Method)
        # We put instructions inside the prompt to avoid the 400 'systemInstruction' error.
        full_prompt = f"""
        INSTRUCTIONS: You are the HITS Official Assistant. 
        - Use ONLY the Context below to answer. 
        - If the answer isn't in the Context, say you don't know and give the email.
        - Use Markdown tables for specs and bullets for fees.
        
        CONTEXT:
        {context}
        
        USER QUESTION:
        {clean_query}
        """

        try:
            # Try the newest model (3.1)
            response = client.models.generate_content(
                model="gemini-3.1-flash", 
                contents=full_prompt
            )
        except Exception as model_error:
            print(f"Model 3.1 failed, falling back to 1.5: {model_error}")
            # SILENT FALLBACK: If 3.1 is missing/busy, use 1.5 so the user sees NO error.
            response = client.models.generate_content(
                model="gemini-1.5-flash", 
                contents=full_prompt
            )
        
        return {"response": response.text}

    except Exception as e:
        print(f"Detailed Server Error: {str(e)}")
        return {"response": "The HITS system is busy. Please contact **info@hindustanuniv.ac.in**."}

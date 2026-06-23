from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
import faiss
import numpy as np
import os

load_dotenv()
app = FastAPI()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

print("Groq connected successfully")
# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
with open("data/docs.txt", "r") as f:
    documents = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]

print(documents)

# Create embeddings
document_embeddings = np.array(
    model.encode(documents),
    dtype=np.float32
)

print(document_embeddings.shape)

# Create FAISS index
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

# Request model
class SearchRequest(BaseModel):
    query: str

# Home route
@app.get("/")
def home():
    return {"message": "Semantic Search API Running"}

# Search route
@app.post("/search")
def semantic_search(request: SearchRequest):

    query_embedding = np.array(
        model.encode([request.query]),
        dtype=np.float32
    )

    distances, indices = index.search(query_embedding, k=2)

    results = [documents[i] for i in indices[0]]

    return {
        "query": request.query,
        "results": results
    }
@app.post("/ask")
def ask_ai(request: SearchRequest):

    try:

        # Convert query → embedding
        query_embedding = np.array(
            model.encode([request.query]),
            dtype=np.float32
        )

        # Search documents
        distances, indices = index.search(
            query_embedding,
            k=2
        )

        # Retrieve context
        context = "\n".join(
            [documents[i] for i in indices[0]]
        )

        # Prompt
        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {request.query}
        """

        # Groq API call
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        return {
            "query": request.query,
            "context": context,
            "answer": answer
        }

    except Exception as e:
        return {
            "error": str(e)
        }
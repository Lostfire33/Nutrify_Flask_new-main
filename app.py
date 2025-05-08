from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from LLM.groq_runtime import GroqRunTime
import chromadb
from typing import List, Dict, Any
from functools import lru_cache

app = FastAPI()

# Load model ringan dan database sekali saat startup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lebih ringan dari mpnet
chroma_client = chromadb.PersistentClient(path="Database")
collection = chroma_client.get_or_create_collection(name="nutrition")

# Request models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatMessage(BaseModel):
    role: str
    content: str

class UserPrompt(BaseModel):
    user_id: str
    user_prompt: str
    previous_chat: List[ChatMessage]

@app.get("/")
def read_root():
    return {"message": "timuca"}

# Cache untuk encoding teks agar tidak encode ulang teks yang sama
@lru_cache(maxsize=256)
def get_cached_embedding(text: str):
    return embedding_model.encode([text]).tolist()[0]

def retrieve_documents(query: str, top_k: int = 5):
    query_embedding = get_cached_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_docs = []
    for metadata_list in results.get("metadatas", []):
        if metadata_list:
            retrieved_docs.extend(metadata_list)
    return retrieved_docs

def sentence_similarity(query_embedding, retrieved_docs):
    if not retrieved_docs:
        return None

    doc_embeddings = embedding_model.encode([doc['name'] for doc in retrieved_docs])
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    return retrieved_docs[best_match_idx]

# Cache hasil ringkasan makanan
@lru_cache(maxsize=128)
def get_summary(text: str):
    groq_run = GroqRunTime()
    system_prompt = (
        "Anda adalah asisten pengambil nama makanan dari kalimat. "
        "Tolong ubah atau ringkas kalimat pengguna agar lebih jelas untuk pencarian makanan. "
        "Hanya respon dengan ringkasan makanan pengguna."
    )
    response = groq_run.generate_response(system_prompt, text)
    return response.choices[0].message.content

@app.post("/search/")
def search_food(request: QueryRequest):
    query_embedding = get_cached_embedding(request.query)
    retrieved_docs = retrieve_documents(request.query, top_k=request.top_k)

    if not retrieved_docs:
        return {"results": []}

    best_match = sentence_similarity(query_embedding, retrieved_docs)
    return {"results": best_match}

@app.post("/chatbot/")
def food_chatbot(request: UserPrompt):
    # Batasi chat history untuk menghemat memori
    max_history = 5
    recent_chat = request.previous_chat[-max_history:]

    chat_history = ""
    for message in recent_chat:
        role = "User" if message.role == "user" else "Asisten"
        chat_history += f"{role}: {message.content}\n"
    chat_history += f"User: {request.user_prompt}\n"

    refined_food = get_summary(chat_history)

    query_embedding = get_cached_embedding(refined_food)
    retrieved_docs = retrieve_documents(refined_food, top_k=3)

    if not retrieved_docs:
        return {"results": []}

    best_match = sentence_similarity(query_embedding, retrieved_docs)

    if not best_match:
        return {"results": []}

    groq_run = GroqRunTime()
    system_prompt = (
        f"Anda adalah asisten pencarian kandungan makanan berbahasa Indonesia. "
        f"Nama anda adalah Calorify. "
        f"Jawab pertanyaan user berdasarkan informasi makanan ini:\n"
        f"Nama makanan: {best_match.get('name', '-')}, "
        f"Kalori: {best_match.get('calories', '-')}, "
        f"Karbohidrat: {best_match.get('carbohydrate', '-')}, "
        f"Lemak: {best_match.get('fat', '-')}, "
        f"Protein: {best_match.get('proteins', '-')}\n\n"
        f"Berikut adalah percakapan sebelumnya:\n{chat_history}"
    )
    response = groq_run.generate_response(system_prompt, request.user_prompt)

    return {
        "results": best_match,
        "refined_food": refined_food,
        "llm_response": response.choices[0].message.content,
        "image_url": best_match.get('image', None)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

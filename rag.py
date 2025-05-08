import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from LLM.groq_runtime import GroqRunTime
from functools import lru_cache

# Load embedding model dan Chroma client sekali saja
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
_chroma_client = chromadb.PersistentClient(path="Database")
_collection = _chroma_client.get_or_create_collection(name="nutrition")

@lru_cache(maxsize=256)
def get_cached_embedding(text: str):
    return _embedding_model.encode([text]).tolist()[0]

class RagChroma:
    def __init__(self):
        self.embedding_model = _embedding_model
        self.collection = _collection
        self.groq = GroqRunTime()

    def retrieve_documents(self, query: str, top_k: int = 5):
        query_embedding = get_cached_embedding(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        retrieved_docs = []
        for metadata_list in results.get("metadatas", []):
            if metadata_list:
                retrieved_docs.extend(metadata_list)
        return retrieved_docs

    def sort_resources(self, query: str, resources: list):
        if not resources:
            return None

        doc_texts = [doc['name'] for doc in resources]
        doc_embeddings = self.embedding_model.encode(doc_texts)
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        most_similar_idx = np.argmax(similarities)
        return resources[most_similar_idx]

    @lru_cache(maxsize=128)
    def get_summary(self, query: str):
        system_prompt = (
            "Anda adalah asisten pengambil nama makanan dari kalimat. "
            "Tolong ubah atau ringkas kalimat pengguna agar lebih jelas untuk pencarian makanan. "
            "Hanya respon dengan nama makanan pengguna."
        )
        response = self.groq.generate_response(system_prompt, query)
        return response.choices[0].message.content.strip()

    def rag_search(self, query: str):
        # 1. Ringkas query
        refined_query = self.get_summary(query)

        # 2. Ambil dokumen dari ChromaDB
        retrieved_docs = self.retrieve_documents(refined_query)

        # 3. Ambil dokumen paling relevan
        best_resource = self.sort_resources(refined_query, retrieved_docs)

        # 4. Jawaban fallback jika data tidak ditemukan
        if not best_resource:
            fallback_response = self.groq.generate_response(
                "Jawablah pertanyaan ini secara umum karena tidak ditemukan di database:", query
            )
            return {
                "best_match": None,
                "refined_query": refined_query,
                "llm_response": fallback_response.choices[0].message.content.strip()
            }

        # 5. Prompt final dengan data makanan
        system_prompt = (
            f"Anda adalah asisten pencarian kandungan makanan berbahasa Indonesia.\n"
            f"Nama anda adalah Calorify. Jawab pertanyaan user berdasarkan informasi makanan ini:\n"
            f"Nama makanan: {best_resource.get('name', '-')}, "
            f"Kalori: {best_resource.get('calories', '-')}, "
            f"Karbohidrat: {best_resource.get('carbohydrate', '-')}, "
            f"Lemak: {best_resource.get('fat', '-')}, "
            f"Protein: {best_resource.get('proteins', '-')}\n"
            f"Jika makanan tidak tersedia, jawab sepengetahuan Anda."
        )
        response = self.groq.generate_response(system_prompt, query)

        return {
            "best_match": best_resource,
            "refined_query": refined_query,
            "llm_response": response.choices[0].message.content.strip()
        }

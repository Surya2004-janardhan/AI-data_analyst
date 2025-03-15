# import os
# import faiss
# import numpy as np
# import pickle
# from groq import Groq

# class RAGWithGroq:
#     def __init__(self, dim=768, index_file="faiss_memory.index"):
#         self.dim = dim
#         self.index_file = index_file
#         self.index = faiss.IndexFlatL2(self.dim)
#         self.memory = []
#         self.client = Groq(api_key=os.environ["GROQ_API_KEY"])  # Initialize Groq Client
        
#         # Load FAISS index if available
#         if os.path.exists(self.index_file):
#             self.load_index()
    
#     def add_memory(self, query, embedding, response):
#         """Stores query-response pair with embeddings in FAISS."""
#         embedding = np.array(embedding).astype('float32').reshape(1, -1)
#         self.index.add(embedding)
#         self.memory.append((query, response))
#         self.save_index()
    
#     def retrieve_memory(self, embedding, top_k=3):
#         """Retrieves relevant past interactions."""
#         embedding = np.array(embedding).astype('float32').reshape(1, -1)
#         if self.index.ntotal == 0:
#             return ""  # No memory yet
        
#         distances, indices = self.index.search(embedding, top_k)
#         retrieved_texts = [self.memory[idx][1] for idx in indices[0] if idx < len(self.memory)]
        
#         return " ".join(retrieved_texts) if retrieved_texts else ""

#     def generate_response(self, query, summary, embedding):
#         """Generates a response using Groq with the correct use of query, summary, and retrieved memory."""
#         retrieved_context = self.retrieve_memory(embedding)

#         # 🔹 **Ensure correct structure of inputs**
#         analysis_prompt = (
#             f"User Question: {query}\n\n"
#             f"Summary of relevant information:\n{summary}\n\n"
#             f"Additional retrieved insights:\n{retrieved_context}\n\n"
#             "Provide a clear, data-driven response based on all the above information."
#         )
        
#         chat_completion = self.client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": summary},
#                 {"role": "system", "content": query},
#                 {"role": "system", "content": "You are an AI data analyst providing data-driven insights."},
#                 {"role": "user", "content": analysis_prompt}

#             ],
#             model="mixtral-8x7b-32768",
#             stream=False
#         )

#         response = chat_completion.choices[0].message.content
#         self.add_memory(query, embedding, response)  # Store response in memory
#         return response

# def rag_pipeline(query, summary, embedding):
#     """
#     Main function to get RAG-based response.
    
#     Inputs:
#     - query (str): The specific user question
#     - summary (str): Pre-provided summary of relevant information
#     - embedding (np.array): Query embedding vector

#     Returns:
#     - AI-generated response (str)
#     """
#     rag = RAGWithGroq()
#     return rag.generate_response(query, summary, embedding)

# # Example Usage
# def main(query, summary):
#     embedding = np.random.rand(768)  # Example embedding
#     response = rag_pipeline(query, summary, embedding)
#     return response

import os
import faiss
import numpy as np
import pickle
from groq import Groq

class RAGWithGroq:
    def __init__(self, dim=768, index_file="faiss_memory.index"):
        self.dim = dim
        self.index_file = index_file
        self.index = faiss.IndexFlatL2(self.dim)
        self.memory = []
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])  # Initialize Groq Client
        
        # Load FAISS index if available
        if os.path.exists(self.index_file):
            self.load_index()
    
    def save_index(self):
        """Saves the FAISS index and memory pairs."""
        faiss.write_index(self.index, self.index_file)
        with open(self.index_file + ".pkl", "wb") as f:
            pickle.dump(self.memory, f)

    def load_index(self):
        """Loads the FAISS index and memory pairs."""
        if os.path.exists(self.index_file) and os.path.exists(self.index_file + ".pkl"):
            self.index = faiss.read_index(self.index_file)
            with open(self.index_file + ".pkl", "rb") as f:
                self.memory = pickle.load(f)
                print(self.memory)

    def add_memory(self, query, embedding, response):
        """Stores query-response pair with embeddings in FAISS."""
        embedding = np.array(embedding).astype('float32').reshape(1, -1)
        self.index.add(embedding)
        self.memory.append((query, response))
        self.save_index()  # ✅ Now the function exists!

    def retrieve_memory(self, embedding, top_k=3):
        """Retrieves relevant past interactions."""
        embedding = np.array(embedding).astype('float32').reshape(1, -1)
        if self.index.ntotal == 0:
            return ""  # No memory yet
        
        distances, indices = self.index.search(embedding, top_k)
        retrieved_texts = [self.memory[idx][1] for idx in indices[0] if idx < len(self.memory)]
        
        return " ".join(retrieved_texts) if retrieved_texts else ""

    def generate_response(self, query, summary, embedding):
        """Generates a response using Groq with the correct use of query, summary, and retrieved memory."""
        retrieved_context = self.retrieve_memory(embedding)

        # 🔹 **Ensure correct structure of inputs**
        analysis_prompt = (
            f"User Question: {query}\n\n"
            f"Summary of relevant information:\n{summary}\n\n"
            f"Additional retrieved insights:\n{retrieved_context}\n\n"
            "Provide a clear, data-driven response based on all the above information."
        )
        
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": summary},
                {"role": "system", "content": query},
                {"role": "system", "content": "You are an AI data analyst providing data-driven insights."},
                {"role": "user", "content": analysis_prompt}

            ],
            model="mixtral-8x7b-32768",
            stream=False
        )

        response = chat_completion.choices[0].message.content
        self.add_memory(query, embedding, response)  # Store response in memory
        return response


def rag_pipeline(query, summary, embedding):
    """
    Main function to get RAG-based response.
    
    Inputs:
    - query (str): The specific user question
    - summary (str): Pre-provided summary of relevant information
    - embedding (np.array): Query embedding vector

    Returns:
    - AI-generated response (str)
    """
    rag = RAGWithGroq()
    return rag.generate_response(query, summary, embedding)

# Example Usage
def main(query, summary):
    embedding = np.random.rand(768)  # Example embedding
    response = rag_pipeline(query, summary, embedding)
    return response

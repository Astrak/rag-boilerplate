from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
import time
import openai
import os
import faiss
import pickle
import numpy as np 

CHECKPOINT_DIR = "polemia-embeddings"

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class Graph:
    def __init__(self, prompt: PromptTemplate):
        self.prompt = prompt
        graph = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph.add_edge(START, "retrieve")
        self.graph = graph.compile()
        self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", temperature=0.1)

    def retrieve(self, state: State):
        print(f'Received question: {state["question"]}')
        MODEL = "text-embedding-3-large"
        response = openai.embeddings.create(input=state["question"],model=MODEL)
        question_embeddings = response.data[0].embedding
        print('Successfully generated embeddings for question')
        matching_documents = self.search_chunked_system(question_embeddings)
        print('Found matching documents')
        return {"context": matching_documents}

    def generate(self, state: State):
        print('############')
        print('############')
        print('############')
        print('############')
        print(f'Received question: {state["question"]}')
        contents: list[str] = []
        for doc in state['context']:
            print(doc.metadata['source'])
            contents.append(f'{doc.page_content}\nAuteur: {doc.metadata["author"]}\nDate: {doc.metadata["date"]}\nSource: {doc.metadata["source"]}\nTitre: {doc.metadata["title"]}')
        print(f'Found {len(contents)} matching documents:')
        docs_content = "\n\n".join(contents)
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        start_time = time.time()
        response = self.llm.invoke(messages)
        delay = time.time() - start_time
        print("LLM answered in %ssec:" % delay)
        print(f"\nRéponse :\n\n{response.content}")
        return {'answer': response.content}
    
    def search_chunked_system(self, query_embedding, results=20):
        """Search across all chunks and merge results"""
        all_results: list[Document] = []
        embeddings_chunks = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('faisschunk_') and f.endswith('.index')]
        n_chunks = len(embeddings_chunks)
        results_per_chunk = results // n_chunks + 1
        for chunk_id in range(n_chunks):
            index = faiss.read_index(f"./{CHECKPOINT_DIR}/faisschunk_{chunk_id}.index")
            scores, indices = index.search(np.array([query_embedding]), results_per_chunk)
            with open(f"./{CHECKPOINT_DIR}/textbatches_{chunk_id}.pkl", "rb") as f:
                chunk_texts: list[Document] = pickle.load(f)
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(chunk_texts):
                    all_results.append(chunk_texts[idx])
                    # all_results.append((score, chunk_texts[idx])) # Tuples list with score
        # all_results.sort(key=lambda x: x[0]) # Sorts tuples list by similarity score
        return all_results
    
    def invoke(self, text):
        return self.graph.invoke({"question": text}) # type: ignore
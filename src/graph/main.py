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



class State(TypedDict):
    question: str
    discussion: str
    context: List[Document]
    answer: str

class Resource(TypedDict):
    url: str
    title: str

class Graph:
    def __init__(self, prompt: PromptTemplate, folders: List[str]):
        self.prompt = prompt
        self.folders = folders
        graph = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph.add_edge(START, "retrieve")
        self.graph = graph.compile()
        self.llm = init_chat_model("grok-3-mini", model_provider="xai", temperature=0.1)

    def retrieve(self, state: State):
        print(f'Received question: {state["question"]}')
        MODEL = "text-embedding-3-large"
        response = openai.embeddings.create(input=state["question"],model=MODEL)
        question_embeddings = response.data[0].embedding
        print('Successfully generated embeddings for question')
        matching_documents = self.search_chunked_system(question_embeddings)
        return {"context": matching_documents}

    def generate(self, state: State):
        print('############')
        print('############')
        print('############')
        print('############')
        print(f'Received question: {state["question"]}')
        context: list[str] = []
        for doc in state['context']:
            context.append(f'{doc.page_content}\nAuteur: {doc.metadata["author"]}\nDate: {doc.metadata["date"]}\nSource: {doc.metadata["source"]}\nTitre: {doc.metadata["title"]}')
        print(f'Found {len(context)} matching documents:')
        str_context = "\n\n".join(context)
        messages = self.prompt.invoke({"question": state["question"], "context": str_context, "discussion": state["discussion"]})
        start_time = time.time()
        response = self.llm.invoke(messages)
        delay = time.time() - start_time
        print("LLM answered in %ssec:" % delay)
        print(f"\nRéponse :\n\n{response.content}")
        return {'answer': response.content, 'context': context  }
    
    def search_chunked_system(self, query_embedding):
        all_results: list[Document] = []
        for folder in self.folders:
            embeddings_folder = folder + 'embeddings/'
            embeddings_chunks = [f for f in os.listdir(embeddings_folder) if f.startswith('faisschunk_') and f.endswith('.index')]
            n_chunks = len(embeddings_chunks)
            results_per_chunk = 8 // n_chunks + 1 # Gather 8 results per source.
            for chunk_id in range(n_chunks):
                index = faiss.read_index(f"{embeddings_folder}faisschunk_{chunk_id}.index")
                scores, indices = index.search(np.array([query_embedding]), results_per_chunk)
                with open(f"{embeddings_folder}textbatches_{chunk_id}.pkl", "rb") as f:
                    chunk_texts: list[Document] = pickle.load(f)
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(chunk_texts):
                        all_results.append((score, chunk_texts[idx]))
        all_results.sort(key=lambda x: x[0]) # Sorts tuples list by similarity score
        for result in all_results:
            print(result[0], result[1].metadata['source'])
        half_index = len(all_results) // 2
        first_half = all_results[:half_index] # Remove the less relevant half relative to the given results (relative filter)
        relevancy_culled_list = [tup for tup in first_half if tup[0] < 1.6] # Remove elements with a dissimilarity superior to 1.6 (absolute filter)
        return [item[1] for item in relevancy_culled_list] 
    
    def invoke(self, question, discussion = ""):
        return self.graph.invoke({"question": question, "discussion": discussion}) # type: ignore
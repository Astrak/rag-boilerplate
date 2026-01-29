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

def gemini_tokens_approx(text: str) -> int:
    return len(text) // 4 + 1 

def gemini_cost_approx(input_text: str, output_text: str) -> float:
    input_tokens  = gemini_tokens_approx(input_text)
    output_tokens = gemini_tokens_approx(output_text)
    input_rate  = 0.1
    output_rate = 0.4
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

class Resource(TypedDict):
    url: str
    title: str

class State(TypedDict):
    question: str
    discussion: str
    context: List[Document]
    answer: str
    resources: List[Resource]
    cost: float

class Graph:
    def __init__(self, prompt: PromptTemplate, folders: List[str]):
        self.prompt = prompt
        self.folders = folders
        graph = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph.add_edge(START, "retrieve")
        self.graph = graph.compile()
        self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", temperature=0.1)

    def retrieve(self, state: State):
        print(f'\033[94mGRAPH: Retrieve: Received question: {state["question"]}')
        MODEL = "text-embedding-3-large"
        response = openai.embeddings.create(input=state["question"],model=MODEL)
        question_embeddings = response.data[0].embedding
        print('\033[94mGRAPH: Retrieve: Successfully generated embeddings for question')
        matching_documents = self.search_embeddings(question_embeddings)
        return {"context": matching_documents}

    def generate(self, state: State):
        context: list[str] = []
        resources: list[Resource] = []
        for doc in state['context']:
            resources.append({'url': doc.metadata['source'], 'title': doc.metadata['title']})
            context.append(f'{doc.page_content}\nAuteur: {doc.metadata["author"]}\nDate: {doc.metadata["date"]}\nSource: {doc.metadata["source"]}\nTitre: {doc.metadata["title"]}')
        str_context = "\n\n".join(context)
        messages = self.prompt.invoke({"question": state["question"], "context": str_context, "discussion": state["discussion"]})
        start_time = time.time()
        response = self.llm.invoke(messages)
        input_text = messages.to_string()
        print(f"\033[94mGRAPH: Full input text to LLM is {len(input_text)} characters long")
        output_text = response.content
        print(f"\033[94mGRAPH: Output text from LLM is {len(output_text)} characters long")
        cost_estimation = gemini_cost_approx(input_text, output_text)
        delay = time.time() - start_time
        print("\033[94mGRAPH: LLM answered in %ssec:" % delay)
        print(f"\033[94mGRAPH: Answer :\n{response.content}")
        return {'answer': response.content, 'context': context, 'resources': resources, 'cost': cost_estimation }
    
    def search_embeddings(self, query_embedding):
        all_results: list[Document] = []
        start_time = time.perf_counter()
        for folder in self.folders:
            folder_start_time = time.perf_counter()
            embeddings_folder = f"./knowledge-sources/{folder}/embeddings/"
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
            print(f'\033[94mGRAPH: Embeddings: {folder}: {(time.perf_counter() - folder_start_time) * 1_000}ms')
        all_results.sort(key=lambda x: x[0]) # Sorts tuples list by similarity score
        # for result in all_results:
        #     print(result[0], result[1].metadata['source'])
        half_index = len(all_results) // 2
        first_half = all_results[:half_index] # Remove the less relevant half relative to the given results (relative filter)
        relevancy_culled_list = [tup for tup in first_half if tup[0] < 1.6] # Remove elements with a dissimilarity superior to 1.6 (absolute filter)
        context = [item[1] for item in relevancy_culled_list] 
        print(f'\033[94mGRAPH: Embeddings: Found {len(context)} matching documents in {(time.perf_counter() - start_time) * 1_000_000}s')
        return context
    
    def invoke(self, question, discussion = ""):
        return self.graph.invoke({"question": question, "discussion": discussion}) # type: ignore
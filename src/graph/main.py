import os
import pickle
import time
from collections.abc import AsyncIterator
from typing import Any, NotRequired, cast

import faiss
import numpy as np
import openai
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.schema import StreamEvent
from langchain_xai import ChatXAI
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from graph.pricing import (
    DEFAULT_EMBEDDING_MODEL,
    CostBreakdown,
    CostDetail,
    chars_to_tokens_approx,
    combine_costs,
    estimate_chat_cost,
    estimate_embedding_cost,
)

CHAT_MODEL = "grok-3"

class Resource(TypedDict):
    url: str
    title: str

class State(TypedDict):
    question: str
    discussion: NotRequired[str]
    folders: NotRequired[list[str]]
    prompt: NotRequired[PromptTemplate]
    context: NotRequired[list[Document]]
    answer: NotRequired[str]
    resources: NotRequired[list[Resource]]
    embedding_cost: NotRequired[CostDetail]
    cost: NotRequired[CostBreakdown]

CONTEXT_CULL = 10

class Graph:
    def __init__(self, prompt: PromptTemplate, folders: list[str]) -> None:
        self.prompt = prompt
        self.folders = folders
        self.preloaded_indices: dict[str, Any] = {}
        self.preloaded_docs: dict[str, list[Document]] = {}
        graph = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph.add_edge(START, "retrieve")
        self.graph = graph.compile()
        self.model_name = CHAT_MODEL
        self.llm = ChatXAI(
            model=self.model_name
        )

    def preload_indices(self) -> None:
        print('Pre-loading indices')
        for folder in self.folders:
            embeddings_folder = f"./knowledge-sources/{folder}/embeddings/"
            for f in os.listdir(embeddings_folder):
                if f.startswith('faisschunk_') and f.endswith('.index'):
                    file = embeddings_folder + f
                    print(f"In folder {folder}, reading file {file}")
                    self.preloaded_indices[file] = faiss.read_index(file)
                    with open(file.replace(".index",'.pkl').replace('faisschunk','textbatches'), "rb") as ff:
                        self.preloaded_docs[file] = pickle.load(ff)

    def retrieve(self, state: State) -> dict[str, Any]:
        print(f'\033[94mGRAPH: Retrieve: Received question: {state["question"]}')
        MODEL = DEFAULT_EMBEDDING_MODEL
        response = openai.embeddings.create(input=state["question"],model=MODEL)
        question_embeddings = response.data[0].embedding
        embedding_usage = getattr(response, "usage", None)
        embedding_tokens = getattr(embedding_usage, "total_tokens", None)
        if embedding_tokens is not None:
            embedding_cost = estimate_embedding_cost(embedding_tokens, MODEL)
        else:
            # OpenAI's embeddings API normally reports usage; fall back to a
            # char-based estimate if a response ever omits it.
            embedding_cost = estimate_embedding_cost(chars_to_tokens_approx(state["question"]), MODEL, estimated=True)
        print('\033[94mGRAPH: Retrieve: Successfully generated embeddings for question')
        folders = state.get("folders", self.folders)
        matching_documents = self.search_embeddings(question_embeddings, folders)
        return {"context": matching_documents, "embedding_cost": embedding_cost}

    async def generate(self, state: State) -> dict[str, Any]:
        context: list[str] = []
        resources: list[Resource] = []
        for doc in state['context']:
            resources.append({'url': doc.metadata['source'], 'title': doc.metadata['title'] or ''})
            context.append(f'{doc.page_content}\nAuteur: {doc.metadata["author"]}\nDate: {doc.metadata["date"]}\nSource: {doc.metadata["source"]}\nTitre: {doc.metadata["title"]}')
        str_context = "\n\n".join(context[:CONTEXT_CULL]) # Cull input to a maximum amount of context elements
        prompt = state.get("prompt", self.prompt)
        messages = prompt.invoke({"question": state["question"], "context": str_context, "discussion": state["discussion"]})
        start_time = time.time()
        response = await self.llm.ainvoke(messages)
        input_text = messages.to_string()
        print(f"\033[94mGRAPH: Full input text to LLM is {len(input_text)} characters long")
        output_text = cast(str, response.content)
        print(f"\033[94mGRAPH: Output text from LLM is {len(output_text)} characters long")
        # Prefer the provider's own token accounting (response.usage_metadata)
        # over the char-based approximation, which is only a fallback for
        # responses/models that don't report usage.
        usage = getattr(response, "usage_metadata", None)
        if usage and usage.get("input_tokens") is not None and usage.get("output_tokens") is not None:
            generation_cost = estimate_chat_cost(usage["input_tokens"], usage["output_tokens"], self.model_name)
        else:
            generation_cost = estimate_chat_cost(
                chars_to_tokens_approx(input_text),
                chars_to_tokens_approx(output_text),
                self.model_name,
                estimated=True,
            )
        cost = combine_costs(state["embedding_cost"], generation_cost)
        delay = time.time() - start_time
        print(f"\033[94mGRAPH: LLM answered in {delay}sec:")
        print(f"\033[94mGRAPH: Answer :\n{response.content}")
        print(f"\033[94mGRAPH: Estimated cost: ${cost['total_cost']:.6f} (embedding=${cost['embedding']['total_cost']:.6f}, generation=${cost['generation']['total_cost']:.6f})")
        return {'answer': response.content, 'context': context, 'resources': resources, 'cost': cost }

    # async def astream_invoke(self, question: str):
    #     initial_state = {"question": question}
    #     async for update in self.graph.astream(initial_state, stream_mode="updates"):
    #         if "generate" in update:
    #             async for delta in update["generate"]:
    #                 yield delta

    def search_embeddings(self, query_embedding: list[float], folders: list[str]) -> list[Document]:
        all_results: list[tuple[float, Document]] = []
        start_time = time.perf_counter()
        results_per_chunk = 4
        quotes: list[tuple[float, Document]] = []
        if not self.preloaded_indices:
            for folder in folders:
                folder_start_time = time.perf_counter()
                embeddings_folder = f"./knowledge-sources/{folder}/embeddings/"
                embeddings_chunks = [f for f in os.listdir(embeddings_folder) if f.startswith('faisschunk_') and f.endswith('.index')]
                n_chunks = len(embeddings_chunks)
                results_per_chunk = 8 // n_chunks + 1 # Gather 8 results per FAISS indx...
                for chunk_id in range(n_chunks):
                    index = faiss.read_index(f"{embeddings_folder}faisschunk_{chunk_id}.index")
                    scores, indices = index.search(np.array([query_embedding]), results_per_chunk)
                    with open(f"{embeddings_folder}textbatches_{chunk_id}.pkl", "rb") as f:
                        chunk_texts: list[Document] = pickle.load(f)
                    for score, idx in zip(scores[0], indices[0]):
                        if idx < len(chunk_texts):
                            all_results.append((score, chunk_texts[idx]))
                print(f'\033[94mGRAPH: Embeddings: {folder}: {((time.perf_counter() - folder_start_time) * 1_000):.0f}ms')
        else:
            filtered_indices = [f for f in self.preloaded_indices if any(f.split('/')[2] in folder for folder in folders)]
            for file in filtered_indices:
                scores, indices = self.preloaded_indices[file].search(np.array([query_embedding]), results_per_chunk)
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.preloaded_docs[file]):
                        if "citations.institut-iliade.com" in file:
                            quotes.append((score, self.preloaded_docs[file][idx]))
                        else:
                            all_results.append((score, self.preloaded_docs[file][idx]))
        all_results.sort(key=lambda x: x[0]) # Sorts tuples list by similarity score
        # for result in all_results:
        #     print(result[0], result[1].metadata['source'])
        half_index = len(all_results) // 2
        first_half = all_results[:half_index] # Remove the less relevant half relative to the given results (relative filter)
        relevancy_culled_list = [tup for tup in first_half if tup[0] < 1.2] # Remove elements with a dissimilarity superior to 1.2 (absolute filter)
        relevancy_culled_list.extend([(score, quote) for score, quote in quotes if score < 1.2]) # Same with quotes
        relevancy_culled_list.sort(key=lambda x: x[0]) # Re-sort with quotes because input sent to the LLM has a further cull and quotes must not be at the bottom.
        context = [item[1] for item in relevancy_culled_list]
        print(f'\033[94mGRAPH: Embeddings: Found {len(context)} matching documents in {((time.perf_counter() - start_time) * 1_000):.0f}ms')
        return context
    
    async def ainvoke(self, question: str, discussion: str = "", folders: list[str] | None = None, prompt: PromptTemplate | None = None) -> dict[str, Any]:
        initial_state: State = {"question": question, "discussion": discussion}
        if folders is not None:
            initial_state["folders"] = folders
        if prompt is not None:
            initial_state["prompt"] = prompt
        return await self.graph.ainvoke(initial_state)

    def astream_events(self, question: str, discussion: str = "", folders: list[str] | None = None, prompt: PromptTemplate | None = None) -> AsyncIterator[StreamEvent]:
        initial_state: State = {"question": question, "discussion": discussion}
        if folders is not None:
            initial_state["folders"] = folders
        if prompt is not None:
            initial_state["prompt"] = prompt
        return self.graph.astream_events(initial_state)
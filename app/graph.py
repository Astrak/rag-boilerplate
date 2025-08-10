from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from scraper import ArticleScraper
import time

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
        self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", temperature=1)

    def retrieve(self, state: State):
        scraper = ArticleScraper(base_url="https://www.polemia.com")
        retrieved_docs = scraper.chunked_similarity_search(state["question"])
        # retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        print('############')
        print('############')
        print('############')
        print('############')
        print(f'Received question: {state["question"]}')
        print(f'Found {len(contents)} matching documents:')
        contents: list[str] = []
        for doc in state['context']:
            print(doc.metadata['source'])
            contents.append(f'{doc.page_content}\nAuteur: {doc.metadata["author"]}\nDate: {doc.metadata["date"]}\nSource: {doc.metadata["source"]}\nTitre: {doc.metadata["title"]}')
        docs_content = "\n\n".join(contents)
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        start_time = time.time()
        response = self.llm.invoke(messages)
        delay = time.time() - start_time
        print("LLM answered in %ssec:" % delay)
        print(f"\nRéponse :\n\n{response.content}")
        return {'answer': response.content}
    
    def invoke(self, text):
        return self.graph.invoke({"question": text}) # type: ignore
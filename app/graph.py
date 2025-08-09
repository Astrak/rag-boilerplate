from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from scraper import ArticleScraper

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

    def retrieve(self, state: State):
        scraper = ArticleScraper(base_url="https://www.polemia.com")
        retrieved_docs = scraper.chunked_similarity_search(state["question"])
        # retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        contents = list[str]
        for doc in state['context']:
            print(doc.metadata)
            try:
                contents.append(f'{doc.page_content}\nAuteur: {doc.metadata["author"]}\nDate: {doc.metadata["date"]}\nSource: {doc.metadata["url"]}\nTitre: {doc.metadata["title"]}')
            except Exception as e:
                print(e, doc.metadata)
        docs_content = "\n\n".join(contents)
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    def invoke(self, text):
        return self.graph.invoke({"question": text}) # type: ignore
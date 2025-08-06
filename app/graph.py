from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class Graph:
    def __init__(self, vector_store: FAISS, prompt: PromptTemplate): 
        self.vector_store = vector_store;
        self.prompt = prompt
        graph = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph.add_edge(START, "retrieve")
        self.graph = graph.compile()

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    def invoke(self, text):
        return self.graph.invoke({"question": text}) # type: ignore
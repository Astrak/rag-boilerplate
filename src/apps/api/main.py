from apps.api.search_prompt import get_search_prompt
from apps.api.search_prompt import get_analyze_prompt
from apps.api.env import fill_env
from graph.main import Graph
from fastapi import FastAPI
from pydantic import BaseModel
import os

folders = os.getenv("FOLDERS")
if not folders:
    raise EnvironmentError("FOLDERS not found. Run with FOLDERS=folder1,folder2,folder3 ...")
folders_list = [item.strip() for item in folders.split(",")]

print('Using following knowledge folders for RAG: ' + ','.join(folders_list))


fill_env()

search_prompt = get_search_prompt()
analyze_prompt = get_analyze_prompt()

search_graph = Graph(search_prompt, folders_list)
analysis_graph = Graph(analyze_prompt, folders_list)

app = FastAPI()

@app.get("/")
def home():
    return "hello world"

class SearchRequest(BaseModel):
    question: str

@app.post("/search")
def search(request: SearchRequest):
    print('search request received: ' + request.question)
    result = search_graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('similarity search finished')
    return {"results": result['answer']}

@app.post("/analyze")
def search(request: SearchRequest):
    print('analyze request received: ' + request.question)
    result = analysis_graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('similarity search finished')
    return {"results": result['answer']}

from apps.api.prompt import get_prompt
from apps.api.env import fill_env
from graph.main import Graph
from fastapi import FastAPI
from pydantic import BaseModel
import os

folders = os.getenv("FOLDERS")
if not folders:
    raise EnvironmentError("FOLDERS not found. Run with FOLDERS=['./my-folder/']")

fill_env()

prompt = get_prompt()

graph = Graph(prompt, folders)

app = FastAPI()

@app.get("/")
def home():
    return "hello world"

class SearchRequest(BaseModel):
    question: str

@app.post("/search")
def search(request: SearchRequest):
    print('search request received: ' + request.question)
    result = graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('similarity search finished')
    return {"results": result['answer']}

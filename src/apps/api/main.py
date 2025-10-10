from apps.api.prompt import get_prompt
from apps.api.env import fill_env
from graph import Graph
from fastapi import FastAPI
from pydantic import BaseModel

fill_env()

prompt = get_prompt()

graph = Graph(prompt)

app = FastAPI()

class SearchRequest(BaseModel):
    question: str

@app.post("/search")
def search(request: SearchRequest):
    print('search request received: ' + request.question)
    result = graph.invoke({"question": request.question})  # pyright: ignore[reportArgumentType]
    print('similarity search finished')
    return {"results": result['answer']}

from apps.api.search_prompt import get_search_prompt
from apps.api.analyze_prompt import get_analyze_prompt
from apps.api.env import fill_env
from graph.main import Graph
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing_extensions import TypedDict
from datetime import datetime, timedelta
import os

origins = [
    "https://polemia.surge.sh",
    "https://ia.polemia.com",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

IP_THROTTLER = {}
COOLDOWN = timedelta(seconds=3)

# @app.middleware("http")
# async def ip_throttle_middleware(request: Request, call_next):
#     ip = request.client.host
#     now = datetime.utcnow()
#     if ip in IP_THROTTLER and now - IP_THROTTLER[ip] < COOLDOWN:
#         return JSONResponse(
#             status_code=409,
#             content={"detail": "Rate limit reached"}
#         )
#     IP_THROTTLER[ip] = now
#     response = await call_next(request)
#     return response

@app.get("/")
def home():
    return "hello world"

class SearchRequest(BaseModel):
    question: str
    sources: list[str]
    llm: str

class Resource(TypedDict):
    url: str
    title: str

@app.post("/search")
def search(request: SearchRequest):
    print('search request received: ' + request.question)
    sources = ",".join([f"./{src}/" for src in request.sources])
    print('sources wanted: ' + sources)
    search_graph.folders = sources.split(',')
    result = search_graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('similarity search finished')
    return {"results": result['answer']}

@app.post("/retrieve")
def search(request: SearchRequest):
    print('search request received: ' + request.question)
    sources = ",".join([f"./{src}/" for src in request.sources])
    print('sources wanted: ' + sources)
    analysis_graph.folders = sources.split(',')
    result = analysis_graph.retrieve({'question': request.question, 'discussion': ''}) 
    resources: list[Resource] = []
    for doc in result['context']:
        resources.append({'url': doc.metadata['source'], 'title': doc.metadata['title']})
    print('similarity search finished')
    return {"resources": resources}

@app.post("/analyze")
def search(request: SearchRequest):
    print('analyze request received: ' + request.question)
    sources = ",".join([f"./{src}/" for src in request.sources])
    print('sources wanted: ' + sources)
    analysis_graph.folders = sources.split(',')
    result = analysis_graph.invoke(request.question)  # pyright: ignore[reportArgumentType]
    print('similarity search finished')
    return {"results": result['answer'], "resources": result['resources'] }

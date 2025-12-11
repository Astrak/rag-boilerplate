from apps.api.search_prompt import get_search_prompt
from apps.api.analyze_prompt import get_analyze_prompt
from apps.api.env import fill_env
from graph.main import Graph
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import TypedDict
from apps.api.ip_throttler_middleware import IPThrottleMiddleware
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

allowed_origins = [
    "https://polemia.surge.sh",
    "https://ia.polemia.com",
]

app = FastAPI()

from typing import Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

@app.middleware("http")
async def log_all_headers_middleware(request: Request, call_next: Callable):
    # Log only OPTIONS requests (or remove the condition to see everything)
    if request.method == "OPTIONS":
        client_ip = request.client.host if request.client else "unknown"
        origin = request.headers.get("origin", "no-origin")
        logger.info("=== FAILED OPTIONS FROM X iOS ===")
        logger.info(f"Client IP: {client_ip}")
        logger.info(f"Origin: {origin}")
        logger.info(f"Access-Control-Request-Method: {request.headers.get('access-control-request-method')}")
        logger.info(f"Access-Control-Request-Headers: {request.headers.get('access-control-request-headers')}")
        logger.info("ALL HEADERS:")
        for name, value in request.headers.items():
            logger.info(f"  {name}: {value}")
        logger.info("=====================================")

    response: Response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=[
        "*",
        "X-Twitter-Active-User",
        "X-Twitter-Client",
        "X-Twitter-Client-Version",
        "X-Twitter-API-Version",
        "X-Twitter-Auth-Type",
        "X-Twitter-Client-DeviceID",
        "X-Twitter-Client-Language",
        "X-Twitter-Response-Format",
    ],
)
app.add_middleware(IPThrottleMiddleware)

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
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost']}

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
    return {"results": result['answer'], "resources": result['resources'], "cost": result['cost'] }

from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import boto3
import os
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

prompt = PromptTemplate.from_template("You are an assistant for question-answering tasks. " +
"Use the following pieces of retrieved context to answer the question." + 
"If you don't know the answer, just say that you don't know. " +
"Use three sentences maximum and keep the answer concise. " +
"""Question: {question} 
Context: {context} 
Answer:""")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found")
os.environ["OPENAI_API_KEY"] = openai_api_key

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if not langsmith_api_key:
    raise EnvironmentError("LANGSMITH_API_KEY not found")
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found")
os.environ["GOOGLE_API_KEY"] = google_api_key

os.makedirs("./vectorstore", exist_ok=True)
s3 = boto3.client('s3', region_name="eu-north-1")
s3.download_file("rag-faiss-index-bucket", "vectorstores/index.faiss", "./vectorstore/index.faiss")
s3.download_file("rag-faiss-index-bucket", "vectorstores/index.pkl", "./vectorstore/index.pkl")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve_from_store1(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve_from_store1, generate])
graph_builder.add_edge(START, "retrieve_from_store1")
graph = graph_builder.compile()

# app = FastAPI()

# class SearchRequest(BaseModel):
#     question: str

# @app.post("/search")
# def search(request: SearchRequest):
#     print('search request received: ' + request.question)
#     result = graph.invoke({"question": request.question})  # pyright: ignore[reportArgumentType]
#     print('similarity search finished')
#     return {"results": result['answer']}

##### Telegram bot

telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
if not telegram_bot_token:
    raise EnvironmentError("TELEGRAM_BOT_TOKEN not found")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Hello {update.effective_user.first_name}! I am your demo bot.") # type: ignore

# Respond to any text message
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Handled")
    result = graph.invoke({"question": update.message.text}) # type: ignore
    print(result)
    await update.message.reply_text(result.answer) # type: ignore

async def main():
    app = ApplicationBuilder().token(telegram_bot_token).build() # type: ignore
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    print("Bot is starting polling now...")
    try:
        await app.run_polling()  # type: ignore
    except Exception as e:
        print(f"Polling failed with exception: {e}")

import asyncio

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.ensure_future(main())
    else:
        loop.run_until_complete(main())

    

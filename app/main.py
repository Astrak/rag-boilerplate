import os
from prompt import get_prompt
from env import fill_env
from graph import Graph
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from scraper import ArticleScraper
from vector_store import get_store
import csv
import boto3
import asyncio

fill_env()

s3 = boto3.client('s3', region_name="eu-north-1")
os.makedirs("./polemia-urls/", exist_ok=True)
s3.download_file("rag-faiss-index-bucket", "polemia-urls/url-list.csv", "./polemia-urls/url-list.csv")
lines: list[str] = []
with open('./polemia-urls/url-list.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        lines.append(row[0])
EXCLUDED_PATHS = ['/mot-clef/', '/page/', '/author/']
scraper = ArticleScraper(base_url="https://www.polemia.com", excluded_paths=EXCLUDED_PATHS)
# articles = scraper.scrape_articles(lines)
# scraper.create_embeddings_with_checkpoint()
# scraper.create_chunked_faiss_system()
# store = scraper.create_vector_store()


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

prompt = get_prompt()

graph = Graph(prompt)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Salutations {update.effective_user.first_name}! Je suis PolemIA, l'IA de Polemia. Je réalise des courtes notes sur vos questions de sociétés en 20 secondes environ. Chaque question est traitée séparément.\n\nQu'est-ce qui vous intéresse ?") # type: ignore

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for _ in range(10): 
        if _ == 0:
            result = graph.invoke(update.message.text)
            answer = result['answer']
        if result:
            break
        await asyncio.sleep(2)
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )
    print(answer)
    await update.message.reply_text(answer, parse_mode="HTML", disable_web_page_preview=True) # type: ignore

bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
app = ApplicationBuilder().token(bot_token).build() # type: ignore
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
print("Bot is starting polling now...")
app.run_polling()  # type: ignore

    

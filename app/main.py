import os
from prompt import get_prompt
from env import fill_env
from graph import Graph
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from scraper import ArticleScraper
from vector_store import get_store
import csv

fill_env()

prompt = get_prompt()

vector_store = get_store()

graph = Graph(vector_store, prompt)

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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Hello {update.effective_user.first_name}! I am your demo bot.") # type: ignore

# Respond to any text message
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Handled")
    # bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
    result = graph.invoke(update.message.text)
    answer = result['answer']
    print(answer)
    await update.message.reply_text(answer) # type: ignore

EXCLUDED_PATHS=[
    "/mot-clef/",
    r".*\.(pdf|jpg|png|gif|css|js)$"  # Regex pattern for file extensions
]

scraper = ArticleScraper(base_url="https://www.polemia.com", excluded_paths=EXCLUDED_PATHS)
article_urls = scraper.discover_urls()
with open("./url_list.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for url in article_urls['discovered']:
        writer.writerow([url])
print(f"Found {len(article_urls['discovered']) + len(article_urls['failed'])} new URLs to scrape")
print('Failed on the following urls:')
print(article_urls['failed'])

bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
app = ApplicationBuilder().token(bot_token).build() # type: ignore
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
print("Bot is starting polling now...")
app.run_polling()  # type: ignore

    

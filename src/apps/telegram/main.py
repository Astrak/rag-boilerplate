import os
from apps.telegram.prompt import get_prompt
from apps.telegram.env import fill_env
from graph.main import Graph
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
import re

folders = os.getenv("FOLDERS")
if not folders:
    raise EnvironmentError("FOLDERS not found. Run with FOLDERS=folder1,folder2,folder3 ...")
folders_list = [item.strip() for item in folders.split(",")]

print('Using following knowledge folders for RAG: ' + ','.join(folders_list))

fill_env()

prompt = get_prompt()

graph = Graph(prompt, folders_list)

sessions = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"👋 Salutations {update.effective_user.first_name}! Je suis PolemIA, l'IA de Polemia.\n\n📑 Je réalise des courtes notes sur vos questions de société. Chaque question est traitée séparément.\n\n👉 Qu'est-ce qui vous intéresse ?") # type: ignore

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        session_id = update.message.from_user.id
        existing_session = sessions.get(session_id)
        if not existing_session:
            result = graph.invoke(update.message.text)
            sessions[session_id] = [update.message.text, result['answer']]
            print('##### RESULTAT :')
            print(result['answer'])
            await update.message.reply_text(result['answer'], parse_mode="HTML", disable_web_page_preview=True)
        else:
            # need_new_retrieval = graph.evaluate(existing_session)
            # if not need_new_retrieval:
            #     # pass to AI
            #     return
            # else:
            discussion = "\n\n".join(existing_session) + update.message.text
            print("##### DISCUSSION COMPLETE")
            print(discussion)
            print('#####')
            result = graph.invoke(discussion)
            sessions[session_id].extend([update.message.text, result['answer']])
            print('##### RESULTAT :')
            print(result['answer'])
            await update.message.reply_text(result['answer'], parse_mode="HTML", disable_web_page_preview=True)

        #### Non-conversational with smileys #####
        # paragraphs = result['answer'].split('\n\n')
        # if len(paragraphs) > 1:
        #     paragraphs[0] = '📝 ' + paragraphs[0]
        # result_with_smileys = re.sub(r'^- ', '👉 ', '\n\n'.join(paragraphs), flags=re.MULTILINE)
        # await update.message.reply_text(result_with_smileys, parse_mode="HTML", disable_web_page_preview=True) 
    except Exception as e:
        print(e)

def main():
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    app = ApplicationBuilder().token(bot_token).build() # type: ignore
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    print("Bot is starting polling now...")
    app.run_polling()  # type: ignore

if __name__ == "__main__":
    main()

    

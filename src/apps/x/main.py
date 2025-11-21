import os
from apps.x.prompt import get_prompt
from src.apps.x.env import fill_env
from graph.main import Graph
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

folders = os.getenv("FOLDERS")
if not folders:
    raise EnvironmentError("FOLDERS not found. Run with FOLDERS=folder1,folder2,folder3 ...")
folders_list = [item.strip() for item in folders.split(",")]

print('Using following knowledge folders for RAG: ' + ','.join(folders_list))

fill_env()

prompt = get_prompt()

graph = Graph(prompt, folders_list)

llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", temperature=0.1)

sessions = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"👋 Salutations {update.effective_user.first_name}! Je suis PolemIA, l'IA de Polemia.\n\n📑 Je réalise des courtes notes sur vos questions de société. Chaque question est traitée séparément.\n\n👉 Qu'est-ce qui vous intéresse ?") # type: ignore

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        session_id = update.message.from_user.id
        existing_session = sessions.get(session_id)
        print(str(sessions[session_id]))
        if not existing_session:
            print('New convo:')
            print(update.message.text)
            needs_rag_yes_no_prompt = PromptTemplate.from_template(
                    """Consigne : Tu es un assistant qui répond uniquement par "oui" ou "non". Pas d'explication, pas d'autre mot, pas de ponctuation, pas de lettres majuscules. Voici le premier message d'un utilisateur d'un assistant IA qui utilise la méthode RAG. Tu dois déterminer si ce message nécessite du RAG, c'est-à-dire de faire une recherche de contexte préalable parmi les documents disponibles de l'assistant, ou si l'utilisateur souhaite juste bavarder et avoir une conversation sans demande de connaissances particulières. S'il y a besoin de trouver des documents spécifiques, réponds "oui", mais s'il souhaite juste avoir une conversation générale, réponds "non".
                
                    Message : {message}

                    Réponse :"""
                    )
            question = needs_rag_yes_no_prompt.invoke({"message": update.message.text})
            answer = llm.invoke(question)
            print("Requires RAG?: ")
            print(answer.content)
            needs_rag = answer.content == "oui"
            result = ""
            if needs_rag:
                print('Run rag')
                answer = graph.invoke(update.message.text)
                result = answer['answer'].strip()
                sessions[session_id] = {'context': answer['context'], 'discussion': [update.message.text, result]}
            else:
                print('Run LLM without RAG')
                answer = llm.invoke("Tu es maintenant PolemIA, une IA créée par Polemia, agissant comme chatbot pour les visiteurs de Polémia (polemia.com et archives.polemia.com), l'Iliade (institut-iliade.com), l'Observatoire de l'Immigration et de la Démographie (OID, observatoire-immigration.fr), l'Observatoire des Décisions de Justice (ODJ, observatoire-justice.fr), Marc Vanguard (marc-vanguard.com) et l'OJIM (ojim.fr), think-tanks français fournissant en essais et rapports les élites françaises. Voici le premier message d'un utilisateur qui te découvre. Réponds-lui, dans la langue de la question, en complétant sa discussion. Sois concis, n'excède pas 50 mots. Voici son message: " + update.message.text)
                print(answer.content)
                result = answer.content.strip()
                sessions[session_id] = {'context': [], 'discussion': [update.message.text, result]}
            await update.message.reply_text(result, parse_mode="HTML", disable_web_page_preview=True)
        else:
            discussion = sessions[session_id]['discussion']
            print('Answering to existing convo')
            needs_new_rag_yes_no_prompt = PromptTemplate.from_template(
                    """Consigne : Tu dois aider un assistant IA avec RAG sur les questions de politique française liées à la gouvernance et à l'immigration, en répondant uniquement par "oui" ou "non" pour déterminer si un nouveau RAG doit être effectué pour répondre aux questions utilisateur. Pas d'explication, pas d'autre mot, pas de ponctuation, pas de lettre majuscule. Voici le nouveau MESSAGE d'un utilisateur. Il est intégré dans une DISCUSSION, et un CONTEXTE de documents RAG déjà compilés. À partir de ces éléments, tu dois déterminer si la réponse que l'assistant devra apporter à ce MESSAGE nécessite des précisions qui n'existent pas dans le CONTEXTE indiqué : dans ce cas, réponds 'oui'. Mais si l'utilisateur souhaite juste bavarder et avoir une conversation sans demande de nouvelles connaissances particulières, ou que la réponse peut être trouvée dans les document existants : réponds 'non'. 
                
                    MESSAGE : 
                    {message}

                    DISCUSSION : 
                    {discussion}

                    CONTEXTE : 
                    {context}

                    Réponse :""")
            question = needs_new_rag_yes_no_prompt.invoke({"message": update.message.text, "discussion": "\n\n".join(discussion), "context": "\n\n".join(sessions[session_id]['context'])})
            answer = llm.invoke(question)
            print("Requires RAG?: ")
            print(answer.content)
            needs_rag = answer.content == "oui"
            result = ""
            if needs_rag:
                print('Run rag')
                answer = graph.invoke(update.message.text)
                result = answer['answer'].strip()
                sessions[session_id] = {'context': answer['context'], 'discussion': sessions[session_id]['discussion'].extend([update.message.text, result])}
            else:
                print('Run LLM without RAG')
                question = prompt.invoke({"question": update.message.text, "discussion": "\n\n".join(discussion), "context": "\n\n".join(sessions[session_id]['context'])})
                answer = llm.invoke(question)
                result = answer.content.strip()
                sessions[session_id] = {'context': sessions[session_id]['context'], 'discussion': sessions[session_id]['discussion'].extend([update.message.text, result])}
                print(result)
            await update.message.reply_text(result, parse_mode="HTML", disable_web_page_preview=True)
    except Exception as e:
        print('Exception: ' + str(e))

def main():
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    app = ApplicationBuilder().token(bot_token).build() # type: ignore
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    print("Bot is starting polling now...")
    app.run_polling()  # type: ignore

if __name__ == "__main__":
    main()

    

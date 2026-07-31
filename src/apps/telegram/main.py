import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from apps.telegram.env import fill_env
from apps.telegram.prompt import get_prompt
from graph.main import Graph

folders = os.getenv("FOLDERS")
if not folders:
    raise OSError("FOLDERS not found. Run with FOLDERS=folder1,folder2,folder3 ...")
folders_list = [item.strip() for item in folders.split(",")]

print('Using following knowledge folders for RAG: ' + ','.join(folders_list))

fill_env()

prompt = get_prompt()

graph = Graph(prompt, folders_list)

llm = init_chat_model("grok-4", model_provider="xai", temperature=0.1)

sessions: dict[int, dict[str, Any]] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.message is not None and update.effective_user is not None
    await update.message.reply_text(f"👋 Salutations {update.effective_user.first_name}! Je suis PolemIA, l'IA de Polemia.\n\n📑 Je réalise des courtes notes sur vos questions de société. Chaque question est traitée séparément.\n\n👉 Qu'est-ce qui vous intéresse ?")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        assert update.message is not None and update.message.from_user is not None and update.message.text is not None
        session_id = update.message.from_user.id
        existing_session = sessions.get(session_id)
        if not existing_session:
            yes_no_prompt = PromptTemplate.from_template(
                    """Consigne : Tu es un assistant qui répond uniquement par "oui" ou "non". Pas d'explication, pas d'autre mot, pas de ponctuation, pas de lettres majuscules. Voici le premier message d'un utilisateur d'un assistant IA qui utilise la méthode RAG. Tu dois déterminer si ce message nécessite du RAG, c'est-à-dire de faire une recherche de contexte préalable parmi les documents disponibles de l'assistant, ou si l'utilisateur souhaite juste bavarder et avoir une conversation sans demande de connaissances particulières. S'il y a besoin de trouver des documents spécifiques, réponds "oui", mais s'il souhaite juste avoir une conversation générale, réponds "non".
                
                Message : {message}

                Réponse :"""
                )
            question = yes_no_prompt.invoke({"message": update.message.text})
            answer = llm.invoke(question)
            needs_rag = answer.content == "oui"
            result: dict[str, Any]
            if needs_rag:
                result = await graph.ainvoke(update.message.text)
            else:
                llm_answer = llm.invoke("Tu es maintenant PolemIA, une IA créée par Polemia, agissant comme chatbot pour les visiteurs de Polémia (polemia.com et archives.polemia.com), l'Iliade (institut-iliade.com), l'Observatoire de l'Immigration et de la Démographie (OID, observatoire-immigration.fr), l'Observatoire des Décisions de Justice (ODJ, observatoire-justice.fr), Marc Vanguard (marc-vanguard.com) et l'OJIM (ojim.fr), think-tanks français fournissant en essais et rapports les élites françaises. Voici le premier message d'un utilisateur qui te découvre. Réponds-lui qui tu es et dis-lui qu'il peut poser des questions précises sur les sujets que traite PolemIA. Réponds lui dans la langue de la question, en complétant sa discussion. Voici son message: " + update.message.text)
                result = {'context': [], 'answer': llm_answer.content}
            sessions[session_id] = {'context': result['context'], 'discussion': [update.message.text, result['answer']]}
            await update.message.reply_text(result['answer'], parse_mode="HTML", disable_web_page_preview=True)
        else:
            # implement flow: use graph or just chat?
            # implement flow: reuse context or retrieve new?
            discussion = existing_session['discussion']
            print("##### DISCUSSION COMPLETE")
            print("\n\n".join(discussion) + update.message.text)
            print('#####')
            result = await graph.ainvoke(update.message.text, "\n\n".join(discussion))
            discussion.extend([update.message.text, result['answer']])
            sessions[session_id] = {'context': result['context'], 'discussion': discussion}
            await update.message.reply_text(result['answer'], parse_mode="HTML", disable_web_page_preview=True)
    except Exception as e:
        print(e)

def main() -> None:
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not bot_token:
        raise OSError("TELEGRAM_BOT_TOKEN not found.")
    app = ApplicationBuilder().token(bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    print("Bot is starting polling now...")
    app.run_polling()

if __name__ == "__main__":
    main()

    

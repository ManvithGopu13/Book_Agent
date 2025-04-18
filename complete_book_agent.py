import os 
from dotenv import load_dotenv
from telegram import Update, Bot
from telegram.ext import Application , MessageHandler, filters, ContextTypes

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OLLAMA_MODEL = "qwen2.5:7b"

from langchain_core.tools import tool
from duckduckgo_search import DDGS
import requests
from PyPDF2 import PdfReader
import tempfile

@tool
def search_and_download_book(book_title: str) -> str:
    """
        Searches for PDF versions of books and returns the first downloadable link.
        Returns PDF path if successful, error message otherwise.
    """

    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(f"{book_title} filetype:pdf", max_results=5)]
            for result in results:
                pdf_url = result.get("href", "")
                if pdf_url.endswith(".pdf"):
                    response = requests.get(pdf_url, timeout= 10)
                    if response.status_code == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                            f.write(response.content)
                            return f.name
                        
        return "PDF not found"
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool
def summarize_pdf(pdf_path: str) -> str:
    """
        Summarizes the content of a PDF file using the local LLM
    """
    try:
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()[:10000]

        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model= OLLAMA_MODEL,
            temperature= 0.7,
            base_url="http://127.0.0.1:11434",
        )

        summary = llm.invoke(
            f"Summarize this book content in 5 paragraphs: {text[:10000]}"
        )
        return summary.content
    except Exception as e:
        return f"Summarization failed : {str(e)}"
    

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def create_agent(tools, system_message):
    """ Helper function to create agents """
    llm = ChatOllama(
        model = OLLAMA_MODEL,
        temperature= 0.7,
        base_url="http://127.0.0.1:11434",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    return AgentExecutor(
        agent=create_tool_calling_agent(
            llm = llm,
            tools= tools,
            prompt= prompt
        ),
        tools= tools
    )


search_agent = create_agent(
    tools = [search_and_download_book],
    system_message="""You are a book search specialist. 
    ALWAYS use the search tool and return ONLY the raw file path string. 
    NEVER add any additional text or formatting."""
)

summary_agent = create_agent(
    tools= [summarize_pdf],
    system_message="You are a book summarization expert. Always use the summary tool."
)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = update.effective_chat.id
    bot = context.bot

    try:
        search_result = await search_agent.ainvoke({"input": user_message})
        file_path = search_result['output'].strip()

        if file_path.startswith("[") and "]" in file_path:
            file_path = file_path.split("](")[1].rstrip(")")

        if os.path.exists(file_path) and file_path.endswith(".pdf"):
            await bot.send_document(
                chat_id= chat_id,
                document= open(file_path, 'rb'),
                caption="Here's the book you requested!"
            )

            summary_result = await summary_agent.ainvoke({
                "input": f"Summarize this book: {file_path}"
            })

            await bot.send_message(
                chat_id= chat_id,
                text= f"Summary: \n\n{summary_result['output']}"
            )

            os.remove(file_path)

        else:
            await bot.send_message(
                chat_id= chat_id,
                text = f"Sorry, Couldn't find the book. Error: {search_result['output']} and recieved file_path: {file_path}"
            )

    except Exception as e:
        await bot.send_message(
            chat_id= chat_id,
            text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    # Start Telegram bot
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Bot is running...")
    application.run_polling()
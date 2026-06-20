from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

def get_llm(provider):

    if provider == "Groq":
        return ChatGroq(
            model="llama-3.1-8b-instant"
        )

    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash"
        )
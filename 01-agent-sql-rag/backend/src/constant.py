import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
DB_PATH = os.getenv("DB_PATH")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

CONFIG = {
    "configurable" : {
        "thread_id": "1234"
    }
}
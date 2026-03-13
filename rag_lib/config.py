import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "DUMMY_API_KEY")
MODEL = os.getenv("MODEL", "z-ai/glm-4.7-flash")
URL_BASE = os.getenv("URL_BASE", "https://api.proxyapi.ru/openrouter/v1")

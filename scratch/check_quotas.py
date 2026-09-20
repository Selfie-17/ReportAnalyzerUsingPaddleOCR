import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

candidates = [
    "gemini-flash-latest",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.6-flash",
]

for m in candidates:
    try:
        resp = client.models.generate_content(
            model=m,
            contents="Say hi"
        )
        print(f"OK [{m}]: {resp.text.strip()}")
    except Exception as e:
        print(f"ERR [{m}]: {e}")

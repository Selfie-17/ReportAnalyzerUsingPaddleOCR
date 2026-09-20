import os
import json
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

models_to_test = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.1-pro-preview"]
for m in models_to_test:
    try:
        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
        resp = client.models.generate_content(
            model=m,
            contents='Return valid JSON: {"model": "' + m + '", "status": "active"}',
            config=config
        )
        print(f"SUCCESS [{m}]: {resp.text.strip()}")
    except Exception as e:
        print(f"FAILED  [{m}]: {e}")

import os
import json
import time
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

def test_call_gemini(prompt: str, api_key: str):
    client = genai.Client(api_key=api_key)
    models = ["gemini-3.6-flash", "gemini-flash-latest"]
    
    for attempt in range(1, 5):
        m = models[0] if attempt <= 2 else models[1]
        try:
            print(f"Calling Gemini ({m}) attempt {attempt}...")
            config = types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
            resp = client.models.generate_content(
                model=m,
                contents=prompt,
                config=config
            )
            data = json.loads(resp.text)
            print("SUCCESS! Keys received:", list(data.keys()))
            return data
        except Exception as e:
            print(f"Attempt {attempt} failed with {m}: {e}")
            time.sleep(attempt * 2)
    return None

api_key = os.environ.get("GEMINI_API_KEY")
res = test_call_gemini('{"instruction": "Return JSON with test=123"}', api_key)
print("Result:", res)

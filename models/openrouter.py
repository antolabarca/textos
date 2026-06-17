import os
from dotenv import load_dotenv
import requests

def generate_with_openrouter(prompt: str, model: str) -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("Falta OPENROUTER_API_KEY en variables de entorno")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 300
        },
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"Error OpenRouter: {response.status_code} - {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]

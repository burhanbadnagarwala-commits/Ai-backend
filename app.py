from flask import Flask, request, jsonify
import os, requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# 🔐 Security
SECRET_TOKEN = os.getenv("AI_BACKEND_TOKEN")

# 🔍 SearXNG
SEARXNG_URL = os.getenv("SEARXNG_URL")

# 🤖 AI keys
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")


# ---------- SEARCH ----------
def searx_search(query):
    try:
        r = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "language": "en"
            },
            timeout=8
        )
        if r.status_code == 200:
            return r.json().get("results", [])[:6]
    except Exception:
        pass
    return []


# ---------- AI PROVIDERS ----------
def call_groq(prompt):
    if not GROQ_KEY:
        return None

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 350
        },
        timeout=12
    )
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    return None


def call_openrouter(prompt):
    if not OPENROUTER_KEY:
        return None

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/mistral-7b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 350
        },
        timeout=12
    )
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    return None


def call_gemini(prompt):
    if not GEMINI_KEY:
        return None

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}",
        json={
            "contents": [{"parts": [{"text": prompt}]}]
        },
        timeout=12
    )
    if r.status_code == 200:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    return None


def call_together(prompt):
    if not TOGETHER_KEY:
        return None

    r = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {TOGETHER_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 350
        },
        timeout=15
    )
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    return None


# ---------- MAIN ROUTE ----------
@app.route("/ai", methods=["POST"])
def ai():
    if request.headers.get("Authorization") != f"Bearer {SECRET_TOKEN}":
        return jsonify({"error": "unauthorized"}), 401

    data = request.json
    query = data.get("query")

    # 🔍 Extra search from SearXNG
    searx_results = searx_search(query)

    context = "\n".join(
        f"- {r.get('title')}: {r.get('content')}"
        for r in searx_results
    )

    prompt = f"""
Answer the query using the sources.
Be neutral, short, and factual.

Query:
{query}

Sources:
{context}
"""

    # 🔁 AI FALLBACK ORDER
    for provider in (
        call_groq,
        call_openrouter,
        call_gemini,
        call_together
    ):
        try:
            answer = provider(prompt)
            if answer:
                return jsonify({
                    "answer": answer.strip(),
                    "provider": provider.__name__,
                    "sources_used": len(searx_results)
                })
        except Exception:
            continue

    return jsonify({"error": "all_ai_failed"}), 503


if __name__ == "__main__":
    app.run()

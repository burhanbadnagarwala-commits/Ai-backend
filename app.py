import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ================= ENV =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
SEARXNG_URL = os.getenv("SEARXNG_URL", "https://searx.party")

# ============== SEARXNG SEARCH =========
def searx_search(query):
    try:
        r = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "language": "en"
            },
            timeout=10
        )
        data = r.json()
        results = []
        for item in data.get("results", [])[:5]:
            results.append(item.get("content", ""))
        return "\n".join(results)
    except Exception:
        return ""

# ============== GROQ ===================
def groq_ai(prompt):
    if not GROQ_API_KEY:
        raise Exception("Groq key missing")

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        },
        timeout=15
    )
    return r.json()["choices"][0]["message"]["content"]

# ============== GEMINI =================
def gemini_ai(prompt):
    if not GEMINI_API_KEY:
        raise Exception("Gemini key missing")

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
        json={
            "contents": [{"parts": [{"text": prompt}]}]
        },
        timeout=15
    )
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

# ============== HUGGINGFACE ============
def hf_ai(prompt):
    if not HF_API_KEY:
        raise Exception("HF key missing")

    r = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers={
            "Authorization": f"Bearer {HF_API_KEY}"
        },
        json={"inputs": prompt},
        timeout=20
    )
    return r.json()[0]["generated_text"]

# ============== MAIN ROUTE ==============
@app.route("/ai-search", methods=["GET"])
def ai_search():
    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    web_context = searx_search(query)

    final_prompt = f"""
Answer the question clearly and simply.
Use the web info if helpful.

Web info:
{web_context}

Question:
{query}
"""

    # Provider fallback chain
    for provider in (groq_ai, gemini_ai, hf_ai):
        try:
            answer = provider(final_prompt)
            return jsonify({
                "query": query,
                "answer": answer
            })
        except Exception:
            continue

    return jsonify({
        "error": "All AI providers are unavailable"
    }), 503

# ============== RUN =====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

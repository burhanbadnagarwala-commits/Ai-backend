import os, io, json, requests
from flask import Flask, request, jsonify, send_file, session, render_template_string
from dotenv import load_dotenv
from fpdf import FPDF
from gtts import gTTS

load_dotenv()

app = Flask(__name__)
app.secret_key = "super-secret-key"  # needed for chat memory

# =========================
# SEARXNG SEARCH
# =========================
def searx_search(query):
    try:
        r = requests.get(
            f"{os.getenv('SEARXNG_URL')}/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=10
        )
        data = r.json()
        return [
            {"title": i["title"], "url": i["url"]}
            for i in data.get("results", [])[:5]
        ]
    except:
        return []


# =========================
# AI PROVIDERS
# =========================
def groq_ai(prompt):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=20
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], "groq"


def gemini_ai(prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={os.getenv('GEMINI_API_KEY')}",
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=20
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"], "gemini"


def hf_ai(prompt):
    r = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        headers={"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"},
        json={"inputs": prompt},
        timeout=30
    )
    r.raise_for_status()
    return r.json()[0]["generated_text"], "huggingface"


def ai_router(prompt):
    for fn in (groq_ai, gemini_ai, hf_ai):
        try:
            return fn(prompt)
        except:
            continue
    return "All AI providers are busy.", "none"


# =========================
# API: CHAT WITH MEMORY
# =========================
@app.route("/ai", methods=["POST"])
def ai():
    q = request.json.get("query")

    history = session.get("history", [])
    search = searx_search(q)

    context = ""
    for h in history[-5:]:
        context += f"User: {h['q']}\nAI: {h['a']}\n"

    context += f"\nUser: {q}\n"
    if search:
        context += "\nSources:\n"
        for s in search:
            context += f"- {s['title']} ({s['url']})\n"

    answer, provider = ai_router(context)

    history.append({"q": q, "a": answer})
    session["history"] = history

    return jsonify({
        "answer": answer,
        "provider": provider,
        "sources": search,
        "history": history
    })


# =========================
# FILE GENERATOR
# =========================
@app.route("/file", methods=["POST"])
def file_gen():
    data = request.json
    ftype = data.get("type", "txt")
    content = data.get("content", "")

    if ftype == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, content)
        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return send_file(buf, download_name="file.pdf", as_attachment=True)

    if ftype == "json":
        buf = io.BytesIO(json.dumps(content, indent=2).encode())
        return send_file(buf, download_name="file.json", as_attachment=True)

    buf = io.BytesIO(content.encode())
    return send_file(buf, download_name="file.txt", as_attachment=True)


# =========================
# FRONTEND (GOOGLE-LIKE UI)
# =========================
@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>AI Search</title>
<style>
body { font-family: Arial; background:#f8f9fa; }
.box { max-width:800px; margin:auto; margin-top:50px; }
input { width:80%; padding:12px; font-size:16px; }
button { padding:12px; }
.chat { background:white; padding:20px; margin-top:20px; border-radius:8px; }
.msg-user { font-weight:bold; }
.msg-ai { margin-top:5px; }
.source { font-size:12px; color:#555; }
</style>
</head>
<body>
<div class="box">
  <h2>🔍 AI Search</h2>
  <input id="q" placeholder="Ask anything..." />
  <button onclick="ask()">Search</button>
  <div id="chat"></div>
</div>

<script>
async function ask(){
  let q = document.getElementById("q").value;
  let r = await fetch("/ai", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({query:q})
  });
  let d = await r.json();
  let chat = document.getElementById("chat");
  chat.innerHTML = "";

  d.history.forEach(h=>{
    chat.innerHTML += `
      <div class="chat">
        <div class="msg-user">You:</div>
        <div>${h.q}</div>
        <div class="msg-ai">AI:</div>
        <div>${h.a}</div>
      </div>`;
  });

  if(d.sources.length){
    chat.innerHTML += "<h4>Sources</h4>";
    d.sources.forEach(s=>{
      chat.innerHTML += `<div class="source"><a href="${s.url}" target="_blank">${s.title}</a></div>`;
    });
  }
}
</script>
</body>
</html>
""")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))

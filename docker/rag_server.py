from fastapi import FastAPI
from pydantic import BaseModel
from tavily import TavilyClient
import requests
import os

app = FastAPI()

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
SGLANG_URL = os.environ.get("SGLANG_URL", "http://sglang:30000")

class Question(BaseModel):
    question: str

def search_minecraft_info(query: str) -> str:
    results = tavily.search(
        query=f"Minecraft {query}",
        search_depth="basic",
        max_results=3,
        include_domains=["minecraft.wiki", "minecraft.fandom.com"]
    )
    context = "\n\n".join([
        f"- {r['content']}" for r in results["results"]
    ])
    return context

def generate_answer(question: str, context: str) -> str:
    payload = {
        "model": "Qwen/Qwen3-8B",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable Minecraft expert assistant. "
                    "Your tone is polite but approachable. "
                    "Use the following reference information to answer accurately:\n\n"
                    f"{context}"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 1024,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    }
    resp = requests.post(f"{SGLANG_URL}/v1/chat/completions", json=payload)
    return resp.json()["choices"][0]["message"]["content"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(body: Question):
    # 1. 웹 검색
    context = search_minecraft_info(body.question)

    # 2. 파인튜닝 모델로 답변 생성
    answer = generate_answer(body.question, context)

    return {
        "question": body.question,
        "context": context,
        "answer": answer
    }
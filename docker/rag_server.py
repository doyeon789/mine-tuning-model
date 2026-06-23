from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tavily import TavilyClient
import requests
import json
import os

app = FastAPI()

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
SGLANG_URL = os.environ.get("SGLANG_URL", "http://sglang:30000")
MODEL_NAME = "ddorin/minecraft-assistant-qwen3-8b"
BLOCKED_RESULT_KEYWORDS = (
    "minecraft dungeons",
    "minecraft legends",
    "minecraft earth",
    "bedrock edition",
    "education edition",
    "adventure chest",
)
DIAMOND_FALSE_CLAIMS = (
    "diamonds form in the nether",
    "diamonds spawn in the nether",
    "find diamonds in the nether",
    "light level detector",
)
NO_CONTEXT_ANSWER = (
    "I could not find reliable Minecraft Java Edition vanilla survival reference "
    "context for that question, so I should not guess."
)

class Question(BaseModel):
    question: str


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(stripped[start:end + 1])


def get_message_text(message: dict) -> str:
    content = message.get("content")
    if content:
        return content

    reasoning_content = message.get("reasoning_content")
    if reasoning_content:
        return reasoning_content

    raise HTTPException(status_code=502, detail="SGLang returned an empty answer")


def validate_answer(question: str, context: str, answer: str) -> dict:
    deterministic_result = deterministic_validation(question, context, answer)
    if deterministic_result is not None:
        return deterministic_result

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict Minecraft Java Edition vanilla survival fact checker. "
                    "Validate the draft answer against the reference context and rules. "
                    "Reject unsupported claims, Minecraft Dungeons, Minecraft Legends, "
                    "Bedrock Edition, Education Edition, and outdated advice unless the user asked for them. "
                    "Return only valid JSON with this schema: "
                    "{\"valid\": boolean, \"issues\": string[], \"corrected_answer\": string}. "
                    "If the draft is valid, keep corrected_answer identical to the draft. "
                    "If the context is insufficient, say so in corrected_answer instead of inventing facts."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Reference context:\n{context}\n\n"
                    f"Draft answer:\n{answer}"
                )
            }
        ],
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = requests.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        validation = extract_json_object(get_message_text(message))
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"SGLang is not ready: {exc}") from exc
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        return {
            "valid": True,
            "issues": ["Validation response could not be parsed."],
            "corrected_answer": answer,
        }

    validation.setdefault("valid", True)
    validation.setdefault("issues", [])
    validation.setdefault("corrected_answer", answer)
    return validation


def is_java_survival_result(result: dict) -> bool:
    text = " ".join(
        [
            result.get("title", ""),
            result.get("url", ""),
            result.get("content", ""),
        ]
    ).lower()
    return not any(keyword in text for keyword in BLOCKED_RESULT_KEYWORDS)


def search_minecraft_info(query: str) -> str:
    search_queries = [
        (
            f"Minecraft Java Edition vanilla survival {query} "
            "-Dungeons -Legends -Bedrock -Education"
        ),
        (
            f"site:minecraft.wiki Minecraft Java Edition vanilla survival {query} "
            "-Dungeons -Legends -Bedrock -Education"
        ),
    ]
    filtered_results = []
    for search_query in search_queries:
        results = tavily.search(
            query=search_query,
            search_depth="basic",
            max_results=6,
            include_domains=["minecraft.wiki"],
        )
        filtered_results = [
            result for result in results.get("results", [])
            if is_java_survival_result(result)
        ][:3]
        if filtered_results:
            break

    context = "\n\n".join([
        f"- {result['content']}" for result in filtered_results
    ])
    return context


def deterministic_validation(question: str, context: str, answer: str) -> dict | None:
    if not context.strip():
        return {
            "valid": False,
            "issues": ["No reference context was retrieved."],
            "corrected_answer": NO_CONTEXT_ANSWER,
        }

    lower_question = question.lower()
    lower_answer = answer.lower()
    if "diamond" in lower_question:
        issues = [
            claim for claim in DIAMOND_FALSE_CLAIMS
            if claim in lower_answer
        ]
        if "nether" in lower_answer and "diamond" in lower_answer:
            issues.append("Diamonds do not generate in the Nether in Java survival.")
        if issues:
            return {
                "valid": False,
                "issues": sorted(set(issues)),
                "corrected_answer": (
                    "For Minecraft Java Edition survival, mine for diamond ore in the Overworld, "
                    "not the Nether. In modern versions, strip mining around Y=-58 or exploring "
                    "deep caves is a common approach. Bring an iron pickaxe or better, watch for "
                    "lava, and use Fortune if you have it."
                ),
            }

    return None

def generate_answer(question: str, context: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable Minecraft expert assistant. "
                    "Your tone is polite but approachable. "
                    "Answer only for Minecraft Java Edition vanilla survival "
                    "unless the user explicitly asks about another edition or spin-off. "
                    "Ignore Minecraft Dungeons, Minecraft Legends, Bedrock Edition, "
                    "Education Edition, and outdated version advice unless directly relevant. "
                    "If the reference information conflicts, prefer current Java Edition survival advice. "
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
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = requests.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"SGLang is not ready: {exc}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Unexpected SGLang response") from exc

    content = message.get("content")
    if content:
        return content

    reasoning_content = message.get("reasoning_content")
    if reasoning_content:
        return reasoning_content

    raise HTTPException(status_code=502, detail="SGLang returned an empty answer")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(body: Question):
    # 1. 웹 검색
    context = search_minecraft_info(body.question)
    if not context.strip():
        validation = deterministic_validation(body.question, context, "")
        return {
            "question": body.question,
            "context": context,
            "answer": validation["corrected_answer"],
            "draft_answer": "",
            "validation": validation,
        }

    # 2. 파인튜닝 모델로 답변 생성
    draft_answer = generate_answer(body.question, context)
    validation = validate_answer(body.question, context, draft_answer)
    answer = validation["corrected_answer"]

    return {
        "question": body.question,
        "context": context,
        "answer": answer,
        "draft_answer": draft_answer,
        "validation": validation,
    }

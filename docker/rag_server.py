from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tavily import TavilyClient
import requests
import json
import os
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse

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
    "april fools",
    "potato dimension",
    "24w14potato",
    "potone",
)
ALLOWED_WIKI_HOSTS = {"minecraft.wiki", "www.minecraft.wiki"}
BLOCKED_WIKI_NAMESPACES = (
    "talk:",
    "user:",
    "file:",
    "category:",
    "template:",
    "module:",
    "special:",
    "help:",
)
PREFERRED_SOURCE_KEYWORDS = (
    "ore",
    "generation",
    "mechanics",
    "trading",
    "enchanting",
    "villager",
    "biome",
    "structure",
    "mob",
)
WEAK_SOURCE_KEYWORDS = (
    "tutorial:",
    "beginner",
    "getting max gear",
)
PASSAGE_PREFERRED_KEYWORDS = (
    "natural generation",
    "obtaining",
    "crafting",
    "usage",
    "spawning",
    "drops",
    "trading",
    "breeding",
    "taming",
    "generation",
    "y=",
    "y level",
    "overworld",
)
PASSAGE_WEAK_KEYWORDS = (
    "history",
    "gallery",
    "trivia",
    "sounds",
    "data values",
    "issues",
    "references",
    "external links",
)
NO_CONTEXT_ANSWER = (
    "I could not find reliable Minecraft Java Edition vanilla survival reference "
    "context for that question, so I should not guess."
)

class Question(BaseModel):
    question: str


class WikiTextParser(HTMLParser):
    BLOCK_TAGS = {"p", "li", "dd", "dt", "h2", "h3", "h4"}
    SKIP_TAGS = {"script", "style", "table", "sup", "math", "figure"}

    def __init__(self):
        super().__init__()
        self.blocks = []
        self.current = []
        self.current_tag = None
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self._flush()
            self.current_tag = tag

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self._flush()
            self.current_tag = None

    def handle_data(self, data):
        if self.skip_depth or not self.current_tag:
            return
        self.current.append(data)

    def _flush(self):
        text = normalize_text(" ".join(self.current))
        if text:
            self.blocks.append(text)
        self.current = []


def normalize_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
                    "Every gameplay claim in corrected_answer must be supported by the reference context. "
                    "Reject unsupported claims, Minecraft Dungeons, Minecraft Legends, "
                    "Bedrock Edition, Education Edition, and outdated advice unless the user asked for them. "
                    "Return only valid JSON with this schema: "
                    "{\"valid\": boolean, \"issues\": string[], \"corrected_answer\": string}. "
                    "If the draft is valid, keep corrected_answer identical to the draft. "
                    "If the context is insufficient or does not support the draft, set valid to false "
                    "and write a corrected_answer using only supported context. "
                    "If no supported answer can be written, say that reliable context was not found."
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
            "valid": False,
            "issues": ["Validation response could not be parsed."],
            "corrected_answer": (
                "I could not validate this answer against the retrieved Minecraft Java Edition "
                "survival context, so I should not present it as reliable."
            ),
        }

    validation.setdefault("valid", True)
    validation.setdefault("issues", [])
    validation.setdefault("corrected_answer", answer)
    return validation


def is_java_survival_result(result: dict) -> bool:
    parsed_url = urlparse(result.get("url", ""))
    host = parsed_url.netloc.lower()
    if host not in ALLOWED_WIKI_HOSTS:
        return False

    article_path = parsed_url.path.removeprefix("/w/").lower()
    if article_path.startswith(BLOCKED_WIKI_NAMESPACES):
        return False

    text = " ".join(
        [
            result.get("title", ""),
            result.get("url", ""),
            result.get("content", ""),
        ]
    ).lower()
    return not any(keyword in text for keyword in BLOCKED_RESULT_KEYWORDS)


def keyword_terms(text: str) -> set[str]:
    normalized = "".join(
        char.lower() if char.isalnum() else " "
        for char in text
    )
    terms = {
        term for term in normalized.split()
        if len(term) > 2 and term not in {"minecraft", "java", "edition", "survival"}
    }
    singular_terms = {
        term[:-1] for term in terms
        if len(term) > 3 and term.endswith("s")
    }
    return terms | singular_terms


def rank_search_result(result: dict, query: str) -> int:
    title = result.get("title", "")
    url = result.get("url", "")
    content = result.get("content", "")
    title_url = f"{title} {url}".lower()
    all_text = f"{title_url} {content}".lower()
    query_terms = keyword_terms(query)

    score = 0
    score += 8 * sum(1 for term in query_terms if term in title_url)
    score += 2 * sum(1 for term in query_terms if term in all_text)
    score += 6 * sum(1 for keyword in PREFERRED_SOURCE_KEYWORDS if keyword in title_url)
    score -= 12 * sum(1 for keyword in WEAK_SOURCE_KEYWORDS if keyword in title_url)

    if "/tutorial:" in url.lower():
        score -= 8
    if "minecraft.wiki/w/" in url.lower():
        score += 4
    return score


def fetch_wiki_blocks(url: str) -> list[str]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "mine-tuning-rag/0.1"},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return []

    parser = WikiTextParser()
    parser.feed(resp.text)
    parser.close()
    return parser.blocks


def split_long_passage(text: str, max_chars: int = 900) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def build_passages(results: list[dict]) -> list[dict]:
    passages = []
    for result in results:
        title = result.get("title", "")
        url = result.get("url", "")
        blocks = fetch_wiki_blocks(url)
        current_section = ""
        for block in blocks:
            if len(block) < 80:
                if 3 <= len(block) <= 70:
                    current_section = block
                continue
            if any(keyword in block.lower() for keyword in BLOCKED_RESULT_KEYWORDS):
                continue
            sectioned_block = (
                f"Section: {current_section}. {block}"
                if current_section else block
            )
            for chunk in split_long_passage(sectioned_block):
                passages.append({
                    "title": title,
                    "url": url,
                    "text": chunk,
                })
    return passages


def rank_passage(passage: dict, question: str, search_query: str) -> float:
    title = passage.get("title", "")
    url = passage.get("url", "")
    text = passage.get("text", "")
    title_url = f"{title} {url}".lower()
    passage_text = text.lower()
    target_terms = keyword_terms(f"{question} {search_query}")

    keyword_score = 0.0
    keyword_score += 8 * sum(1 for term in target_terms if term in title_url)
    keyword_score += 3 * sum(1 for term in target_terms if term in passage_text)
    keyword_score += 5 * sum(1 for keyword in PASSAGE_PREFERRED_KEYWORDS if keyword in passage_text)
    keyword_score -= 8 * sum(1 for keyword in PASSAGE_WEAK_KEYWORDS if keyword in passage_text[:120].lower())

    passage_terms = keyword_terms(text)
    overlap = len(target_terms & passage_terms)
    semantic_like_score = overlap / max(len(target_terms), 1)

    return keyword_score + (semantic_like_score * 10)


def select_passages(
    question: str,
    search_query: str,
    results: list[dict],
    max_passages: int = 6,
) -> list[dict]:
    passages = build_passages(results)
    ranked = sorted(
        passages,
        key=lambda passage: rank_passage(passage, question, search_query),
        reverse=True,
    )
    return ranked[:max_passages]


def rewrite_search_query(question: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question into one concise web search query for source documents. "
                    "Target only Minecraft Java Edition vanilla survival. "
                    "Exclude Minecraft Dungeons, Minecraft Legends, Bedrock Edition, and Education Edition. "
                    "Prefer current Java Edition information. "
                    "Keep important item, mob, biome, structure, version, and mechanic names. "
                    "Prefer Minecraft Wiki source-document terms such as block, item, mob, biome, "
                    "structure, mechanics, generation, loot, trading, or enchanting. "
                    "Avoid broad tutorial or beginner-guide wording unless the user asks for a tutorial. "
                    "Return only the search query. Do not answer the question."
                )
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": 80,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = requests.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        rewritten_query = get_message_text(data["choices"][0]["message"])
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return question

    rewritten_query = rewritten_query.strip().strip('"').strip("'")
    rewritten_query = " ".join(rewritten_query.split())
    return rewritten_query or question


def search_minecraft_info(query: str) -> dict:
    rewritten_query = rewrite_search_query(query)
    search_queries = [
        (
            f"site:minecraft.wiki/w {rewritten_query} "
            "-Dungeons -Legends -Bedrock -Education"
        ),
        (
            f"minecraft.wiki {rewritten_query} Java Edition mechanics generation "
            "-Dungeons -Legends -Bedrock -Education"
        ),
        rewritten_query,
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
        ]
        filtered_results = sorted(
            filtered_results,
            key=lambda result: rank_search_result(result, rewritten_query),
            reverse=True,
        )[:3]
        if filtered_results:
            break

    selected_passages = select_passages(query, rewritten_query, filtered_results)
    if selected_passages:
        context = "\n\n".join([
            "\n".join([
                f"Source: {passage.get('title', '')}",
                f"URL: {passage.get('url', '')}",
                f"Passage: {passage.get('text', '')}",
            ])
            for passage in selected_passages
        ])
    else:
        context = "\n\n".join([
            "\n".join([
                f"Source: {result.get('title', '')}",
                f"URL: {result.get('url', '')}",
                f"Content: {result.get('content', '')}",
            ])
            for result in filtered_results
        ])

    return {
        "query": rewritten_query,
        "context": context,
        "results": [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
            }
            for result in filtered_results
        ],
        "passages": [
            {
                "title": passage.get("title", ""),
                "url": passage.get("url", ""),
                "text": passage.get("text", ""),
            }
            for passage in selected_passages
        ],
    }


def deterministic_validation(question: str, context: str, answer: str) -> dict | None:
    if not context.strip():
        return {
            "valid": False,
            "issues": ["No reference context was retrieved."],
            "corrected_answer": NO_CONTEXT_ANSWER,
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
                    "Base the answer on the reference information. "
                    "If the reference information is insufficient, say that reliable context was not found "
                    "instead of guessing. "
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
    search = search_minecraft_info(body.question)
    context = search["context"]
    if not context.strip():
        validation = deterministic_validation(body.question, context, "")
        return {
            "question": body.question,
            "search_query": search["query"],
            "search_results": search["results"],
            "retrieved_passages": search["passages"],
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
        "search_query": search["query"],
        "search_results": search["results"],
        "retrieved_passages": search["passages"],
        "context": context,
        "answer": answer,
        "draft_answer": draft_answer,
        "validation": validation,
    }

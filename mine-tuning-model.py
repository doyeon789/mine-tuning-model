"""
mine-tuning-model - Windows 로컬 버전
Google Colab 의존성 제거, 윈도우 데스크탑에서 바로 실행 가능
"""

# ───────────────────────────────────────────────
# 0. 필수 라이브러리 설치 (처음 실행 시만)
# ───────────────────────────────────────────────
# 아래 명령어를 터미널에서 먼저 실행하세요:
#
#   pip install chromadb langchain langchain-community langchain-openai \
#               sentence-transformers transformers datasets gradio \
#               langgraph torch accelerate -q
#
# GPU(CUDA)가 있다면 torch를 따로 설치:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# ───────────────────────────────────────────────
# 1. 경로 설정
# ───────────────────────────────────────────────
from pathlib import Path

BASE_DIR   = Path(__file__).parent          # 이 .py 파일이 있는 폴더
CHROMA_DIR = BASE_DIR / "chroma_db"        # ChromaDB 영구 저장 경로
CHROMA_DIR.mkdir(exist_ok=True)

print(f"작업 경로     : {BASE_DIR}")
print(f"ChromaDB 경로 : {CHROMA_DIR}")

# ───────────────────────────────────────────────
# 2. 데이터 로드
# ───────────────────────────────────────────────
from datasets import load_dataset

print("\n[데이터 로드 중...]")
ds = load_dataset("lparkourer10/minecraft-wiki")
print(ds)

# 샘플 3개 출력
for i in range(3):
    print(f"\n=== 샘플 {i+1} ===")
    print(f"URL : {ds['train']['url'][i]}")
    print(f"Q   : {ds['train']['question'][i]}")
    print(f"A   : {ds['train']['answer'][i][:200]}")

# ───────────────────────────────────────────────
# 3. 임베딩 모델 로드
# ───────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

print("\n[임베딩 모델 로드 중...]")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"임베딩 차원: {embedding_model.get_sentence_embedding_dimension()}")

# ───────────────────────────────────────────────
# 4. ChromaDB 구축 (이미 있으면 스킵)
# ───────────────────────────────────────────────
import chromadb

chroma_client   = chromadb.PersistentClient(path=str(CHROMA_DIR))
COLLECTION_NAME = "minecraft_rag"
SAMPLE_SIZE     = 5000

existing = [c.name for c in chroma_client.list_collections()]
if COLLECTION_NAME in existing:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"\n기존 컬렉션 로드 완료: {collection.count()}개 데이터")
else:
    print("\n[ChromaDB 구축 중...]")
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    answers   = ds["train"]["answer"][:SAMPLE_SIZE]
    questions = ds["train"]["question"][:SAMPLE_SIZE]
    urls      = ds["train"]["url"][:SAMPLE_SIZE]

    batch_size = 500
    for i in range(0, SAMPLE_SIZE, batch_size):
        batch_answers = answers[i:i+batch_size]
        embeddings = embedding_model.encode(batch_answers, show_progress_bar=True).tolist()
        collection.add(
            documents=batch_answers,
            embeddings=embeddings,
            metadatas=[{"question": q, "url": u}
                       for q, u in zip(questions[i:i+batch_size], urls[i:i+batch_size])],
            ids=[str(j) for j in range(i, i+len(batch_answers))]
        )
        print(f"  {min(i+batch_size, SAMPLE_SIZE)}/{SAMPLE_SIZE} 완료")

    print(f"\n총 {collection.count()}개 저장됨")

# ───────────────────────────────────────────────
# 5. 검색 함수
# ───────────────────────────────────────────────
def retrieve(query: str, top_k: int = 3):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0], results["metadatas"][0]

# 검색 테스트
print("\n[검색 테스트]")
docs, metas = retrieve("How to find diamonds?")
for i, (doc, meta) in enumerate(zip(docs, metas)):
    print(f"=== 검색 결과 {i+1} ===")
    print(f"URL    : {meta['url']}")
    print(f"Q      : {meta['question']}")
    print(f"Answer : {doc[:200]}\n")

# ───────────────────────────────────────────────
# 6. LLM 로드 (GPU/CPU 자동 감지)
# ───────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

if torch.cuda.is_available():
    dtype      = torch.float16
    device_arg = {"device_map": "auto"}
    pipe_device = 0
    print(f"\nGPU 사용: {torch.cuda.get_device_name(0)}")
else:
    dtype      = torch.float32
    device_arg = {}
    pipe_device = -1
    print("\nCPU 사용 (속도가 느릴 수 있습니다)")

print("[LLM 로드 중...]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, **device_arg)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    device=pipe_device,
)
print("LLM 로드 완료")

# ───────────────────────────────────────────────
# 7. RAG 파이프라인
# ───────────────────────────────────────────────
def rag_answer(query: str) -> str:
    docs, metas = retrieve(query, top_k=3)
    context = "\n\n".join(docs)

    prompt = f"""You are a helpful Minecraft expert assistant.
Use the following context to answer the question accurately.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
Answer:"""

    messages = [{"role": "user", "content": prompt}]
    result   = llm(messages)
    answer   = result[0]["generated_text"][-1]["content"]

    print(f"출처: {metas[0]['url']}\n")
    return answer

# RAG 테스트
print("\n[RAG 테스트]")
response = rag_answer("How to find diamonds?")
print(response)

# ───────────────────────────────────────────────
# 8. Gradio UI 실행
# ───────────────────────────────────────────────
import gradio as gr

def chat(message, history):
    return rag_answer(message)

demo = gr.ChatInterface(
    fn=chat,
    title="⛏️ Minecraft Guide LLM",
    description="Minecraft Wiki 기반 AI 가이드에게 무엇이든 물어보세요!",
    examples=[
        "How to find diamonds?",
        "How do I defeat the Ender Dragon?",
        "How to make a Nether portal?"
    ],
)

# 로컬: http://localhost:7860 접속
# 외부 공유가 필요하면 share=True 로 변경
demo.launch(share=False)
"""
mine-tuning-model - Windows 로컬 버전 (함수화)

실행 전 설치:
  pip install chromadb sentence-transformers transformers datasets gradio torch accelerate -q

GPU(CUDA)가 있다면:
  pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import torch
import chromadb
import shutil
import gradio as gr

# ───────────────────────────────────────────────
# 설정값
# ───────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CHROMA_DIR      = BASE_DIR / "chroma_db"
COLLECTION_NAME = "minecraft_rag"
SAMPLE_SIZE     = 5000
MODEL_ID        = "Qwen/Qwen2.5-1.5B-Instruct"


# ───────────────────────────────────────────────
# 함수 정의
# ───────────────────────────────────────────────
def load_data():
    print("[데이터 로드 중...]")
    ds = load_dataset("lparkourer10/minecraft-wiki")
    print(ds)
    return ds


def load_embedding_model():
    print("[임베딩 모델 로드 중...]")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"임베딩 차원: {model.get_embedding_dimension()}")
    return model


def build_or_load_chroma_test(embedding_model, ds):
    CHROMA_DIR.mkdir(exist_ok=True)

    # 기존 chroma_db 삭제 (Windows 호환)
    if CHROMA_DIR.exists():
        print(f"기존 chroma_db 삭제 중...")
        import gc
        gc.collect()  # 가비지 컬렉션으로 파일 핸들 해제
        try:
            shutil.rmtree(CHROMA_DIR)
        except PermissionError:
            # 삭제 안되면 그냥 기존 컬렉션 재사용
            print("기존 chroma_db 재사용 중...")
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            try:
                client.delete_collection(name=COLLECTION_NAME)
            except:
                pass
            collection = client.create_collection(name=COLLECTION_NAME)
            # 아래 임베딩 구축 코드로 이어서 실행
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
            print(f"총 {collection.count()}개 저장됨")
            return collection

    # 정상 삭제됐을 때
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.create_collection(name=COLLECTION_NAME)

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

    print(f"총 {collection.count()}개 저장됨")
    return collection

def build_or_load_chroma(embedding_model, ds):
    CHROMA_DIR.mkdir(exist_ok=True)

    # 기존 chroma_db 삭제
    if CHROMA_DIR.exists():
        print(f"기존 chroma_db 삭제 중...")
        import gc
        gc.collect()
        try:
            shutil.rmtree(CHROMA_DIR)
        except PermissionError:
            print("⚠️ 삭제 실패 — 탐색기에서 chroma_db 폴더를 직접 삭제 후 재실행해주세요.")
            raise

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.create_collection(name=COLLECTION_NAME)

    # 전체 데이터
    answers   = ds["train"]["answer"]
    questions = ds["train"]["question"]
    urls      = ds["train"]["url"]
    total     = len(answers)
    print(f"[ChromaDB 전체 구축 중... 총 {total}개]")

    batch_size = 500
    for i in tqdm(range(0, total, batch_size), desc="벡터 DB 구축"):
        batch_answers   = answers[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        batch_urls      = urls[i:i+batch_size]

        embeddings = embedding_model.encode(
            batch_answers,
            show_progress_bar=False
        ).tolist()

        collection.add(
            documents=batch_answers,
            embeddings=embeddings,
            metadatas=[{"question": q, "url": u} for q, u in zip(batch_questions, batch_urls)],
            ids=[str(j) for j in range(i, i+len(batch_answers))]
        )

    print(f"🎉 완료! 총 {collection.count()}개 저장됨")
    return collection

def load_llm():
    if torch.cuda.is_available():
        dtype       = torch.float16
        device_arg  = {"device_map": "auto"}
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        dtype       = torch.float32
        device_arg  = {}
        print("CPU 사용 (속도가 느릴 수 있습니다)")

    print("[LLM 로드 중...]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, **device_arg)
    llm       = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
    )
    print("LLM 로드 완료")
    return llm


def retrieve(query: str, embedding_model, collection, top_k: int = 3):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0], results["metadatas"][0]


def rag_answer(query: str, embedding_model, collection, llm) -> str:
    docs, metas = retrieve(query, embedding_model, collection)
    context = "\n\n".join(docs)

    prompt = f"""You are a helpful Minecraft expert assistant.
Use the following context to answer the question accurately.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
Answer:"""

    result = llm([{"role": "user", "content": prompt}])
    answer = result[0]["generated_text"][-1]["content"]
    print(f"출처: {metas[0]['url']}\n")
    return answer


def launch_ui(embedding_model, collection, llm):
    def chat(message, history):
        return rag_answer(message, embedding_model, collection, llm)

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
    demo.launch(share=False)  # 외부 공유: share=True


# ───────────────────────────────────────────────
# main
# ───────────────────────────────────────────────
def main():
    print(f"작업 경로     : {BASE_DIR}")
    print(f"ChromaDB 경로 : {CHROMA_DIR}\n")

    ds              = load_data()
    embedding_model = load_embedding_model()
    collection      = build_or_load_chroma_test(embedding_model, ds)
    # collection      = build_or_load_chroma(embedding_model, ds)
    llm             = load_llm()

    # 동작 확인용 테스트
    print("\n[RAG 테스트]")
    print(rag_answer("How to find diamonds?", embedding_model, collection, llm))

    # Gradio UI 실행 → http://localhost:7860
    launch_ui(embedding_model, collection, llm)


if __name__ == "__main__":
    main()
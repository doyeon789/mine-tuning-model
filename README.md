# mine-tuning-model

Minecraft Java Edition vanilla survival 질문에 답하기 위한 RAG 서버와 파인튜닝 모델 실행 환경입니다.

이 레포지토리는 사용자의 질문을 Minecraft Wiki 검색용 질의로 바꾸고, Tavily로 관련 문서를 찾은 뒤, 실제 wiki 본문 passage를 다시 랭킹해서 SGLang 모델에 전달합니다. 최종 답변은 근거 검증 단계를 거쳐 초보자도 따라 하기 쉬운 톤으로 반환합니다.

## 주요 기능

- Minecraft Java Edition 바닐라 야생 기준 답변
- Minecraft Dungeons, Legends, Bedrock, Education 등 다른 문맥 필터링
- Tavily 검색 결과에서 `minecraft.wiki` 본문 fetch
- passage chunking, overlap, keyword ranking, RRF rerank
- 선택형 embedding rerank
- SGLang OpenAI-compatible API로 답변 생성과 검증
- 후속 질문을 위한 선택형 대화 히스토리 입력
- ngrok을 통한 외부 공개 URL 연결

## 구조

```text
.
├── docker/
│   ├── docker-compose.yml      # sglang, rag, ngrok 서비스
│   ├── Dockerfile.rag          # FastAPI RAG 서버 이미지
│   └── rag_server.py           # 검색, passage ranking, 답변 생성 API
├── docs/
│   └── rag_server_history.md   # rag_server.py 개선 히스토리
├── data/
│   └── persona.jsonl           # 학습/튜닝용 데이터
├── colab_model_code/           # 모델 학습/실험 노트북
├── requirements-ft.txt         # fine-tuning 관련 패키지 목록
└── log.md                      # 실험 메모
```

## 실행 환경

권장 환경:

- Docker 및 Docker Compose
- NVIDIA GPU
- NVIDIA Container Toolkit
- Hugging Face token
- Tavily API key
- ngrok authtoken

현재 compose는 SGLang 컨테이너에서 다음 모델을 실행합니다.

```text
ddorin/minecraft-assistant-qwen3-8b
```

## 환경 변수

`docker/.env` 파일을 만들고 아래 값을 채웁니다.

```env
HF_TOKEN=your_huggingface_token
TAVILY_API_KEY=your_tavily_api_key
NGROK_AUTHTOKEN=your_ngrok_authtoken

# Optional embedding rerank
EMBEDDING_URL=http://sglang:30000
EMBEDDING_MODEL=
EMBEDDING_CANDIDATE_LIMIT=30
EMBEDDING_WEIGHT=35
```

`EMBEDDING_MODEL`이 비어 있으면 semantic rerank는 꺼지고 keyword/source 기반 RRF rerank만 사용합니다.

## 실행

루트 디렉터리에서 실행합니다.

```bash
docker compose -f docker/docker-compose.yml up --build
```

서비스:

- RAG API: `http://localhost:8000`
- SGLang API: `http://localhost:30000`
- ngrok dashboard: `http://localhost:4040`

상태 확인:

```bash
curl http://localhost:8000/health
```

응답:

```json
{"status":"ok"}
```

## API

### POST `/chat`

기본 요청:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"How do I get sticks?\"}"
```

대화 히스토리를 포함한 요청:

```json
{
  "question": "then, how do I get sticks?",
  "history": [
    {
      "role": "user",
      "content": "I just started this game. What should I do first?"
    },
    {
      "role": "assistant",
      "content": "First, punch a tree to collect logs. Then turn logs into planks and make a crafting table."
    }
  ]
}
```

응답에는 최종 답변뿐 아니라 검색 질의, 검색 결과, retrieved passages, draft answer, validation 결과가 포함됩니다.

주요 필드:

```json
{
  "question": "then, how do I get sticks?",
  "search_query": "Minecraft Wiki sticks crafting planks Java Edition",
  "answer": "First, turn logs into planks in your inventory crafting grid. Then place two planks vertically to craft sticks. Sticks are useful because many early tools need them.",
  "draft_answer": "...",
  "validation": {
    "valid": true,
    "issues": [],
    "corrected_answer": "..."
  }
}
```

## RAG 흐름

1. 사용자 질문 수신
2. 대화 히스토리가 있으면 후속 질문 문맥 정리
3. SGLang으로 Minecraft Wiki 검색용 질의 재작성
4. Tavily로 `minecraft.wiki` 후보 검색
5. Java Edition 바닐라 야생과 맞지 않는 URL 제거
6. wiki 본문 HTML fetch
7. 본문을 passage로 분할하고 긴 passage는 overlap chunk로 재분할
8. keyword/source/semantic 순위를 계산
9. RRF로 최종 passage ranking
10. 선택된 passage를 context로 구성
11. SGLang으로 초안 답변 생성
12. 검증 프롬프트로 unsupported claim과 off-intent detail 제거
13. 최종 `answer` 반환

## 답변 스타일

답변은 초보 Minecraft Java Edition survival 플레이어를 기준으로 작성합니다.

- 따뜻하고 차분한 톤
- 보통 3-5개의 짧은 문장
- `First`, `Then`, `After that` 같은 순서 설명
- 질문과 직접 관련 없는 장비 최적화, 인챈트, 드롭률, 고급 팁 제외
- 3x3 crafting recipe는 슬롯 번호로 설명

예시:

```text
Use a 3x3 crafting table. Put iron ingots in slots 1, 2, and 3, and sticks in slots 5 and 8. Leave the other slots empty.
```

## 개발 메모

문법 검사:

```bash
python -m py_compile docker/rag_server.py
```

서비스 목록 확인:

```bash
docker compose -f docker/docker-compose.yml config --services
```

주의: `docker compose config`는 `.env` 값을 펼쳐서 보여줄 수 있으므로, 로그나 문서에 공유할 때는 secrets를 노출하지 않도록 조심합니다.

## 문서

`docs/rag_server_history.md`에는 `rag_server.py`가 어떤 문제를 해결하면서 현재 구조로 바뀌었는지 단계별로 정리되어 있습니다.

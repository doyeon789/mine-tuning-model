# rag_server.py 변천사와 답변 결과 정리

이 문서는 `docker/rag_server.py`가 처음 만들어진 뒤 현재 구조까지 바뀐 흐름을 정리한다.

답변 예시는 전체 API 응답이 아니라 최종 `answer` 값만 기록한다. `question`, `context`, `draft_answer`, `validation` 전체 JSON은 제외한다.

## 목표

- Minecraft Java Edition 바닐라 야생 기준으로 답변한다.
- Minecraft Dungeons, Minecraft Legends, Bedrock Edition, Education Edition 문맥을 제외한다.
- 검색 결과 스니펫만 믿지 않고 Minecraft Wiki 본문 근거를 사용한다.
- 근거가 부족하면 추측하지 않는다.
- 사용자의 질문 의도에 직접 필요한 내용만 답한다.

## 주요 변천사

| 단계 | 커밋 | 변경 내용 | 해결하려던 문제 |
| --- | --- | --- | --- |
| 1 | `03928d8` | FastAPI RAG 서버 생성, Tavily 검색, SGLang 호출 추가 | 검색 기반 답변 API의 기본 구조 생성 |
| 2 | `ffe8431` | 답변 grounding, SGLang 응답 파싱, 검증 단계 추가 | `answer: null`, 근거 없는 답변, 서버 오류 대응 |
| 3 | `5d15371` | 검색어 재작성 단계 추가 | 사용자의 짧은 질문을 Minecraft Wiki 검색에 맞는 질의로 바꾸기 |
| 4 | `74d08e0` | 검색 결과 랭킹 개선 | 튜토리얼/초보자 가이드보다 원문 문서 우선 |
| 5 | `e42f833` | wiki 소스 필터 강화 | Hypixel SkyBlock, Talk 문서, April Fools 문서 등 오염 결과 제거 |
| 6 | `380b35b` | Minecraft Wiki 페이지 본문 fetch, passage 분할, passage 랭킹 추가 | 검색 스니펫에 필요한 정보가 빠지는 문제 해결 |
| 7 | `cef4c53` | hybrid retriever 구조 추가, 임베딩 rerank 옵션 추가 | 키워드 랭킹과 semantic similarity를 함께 사용할 수 있게 준비 |
| 8 | `54cfae3` | passage chunk overlap 추가 | 긴 본문을 자를 때 핵심 문맥이 앞뒤로 끊기는 문제 완화 |
| 9 | `5221bd8` | RRF rerank 추가 | keyword, source, semantic 순위를 안정적으로 결합 |
| 10 | `6b056f8` | rag upgrade PR merge | 최신 retriever 구조와 히스토리 문서를 main에 반영 |
| 11 | `b46874d` | 초보 친화 프롬프팅 강화 | 직접 의도에 맞는 짧은 답변과 제작법 슬롯 설명을 안정화 |

## 답변 변화

### 초기 동작 실패

처음에는 SGLang 호출과 응답 파싱이 안정적이지 않아 최종 답변이 비어 있었다.

최종 답변:

```text
null
```

### 근거 없는 추측 발생

검색 문맥이 부족하거나 부정확할 때 모델이 Minecraft Java Edition 야생 기준과 맞지 않는 내용을 섞었다.

최종 답변:

```text
dig in and around mountain ranges, diamonds form in the nether too - strip mine at y -58 is a popular spot
```

문제:

- Nether에서 다이아몬드가 생성된다는 잘못된 내용이 포함됐다.
- 질문은 다이아몬드 위치를 묻는데, 근거 없는 일반 팁이 섞였다.

### 근거 없을 때 추측 방지

검색 context가 비어 있으면 답변을 생성하지 않고 거절하도록 바꿨다.

최종 답변:

```text
I could not find reliable Minecraft Java Edition vanilla survival reference context for that question, so I should not guess.
```

효과:

- 잘못된 답변을 만드는 것보다, 근거 부족을 명확히 알리게 됐다.

### 검색어 재작성 후 개선

짧은 질문을 Minecraft Wiki 검색용 질의로 바꿔 검색 성공률을 높였다.

최종 답변:

```text
diamonds are in caves and can also be found with fortune on a pickaxe - strip mining is reliable
```

문제:

- 답변은 나아졌지만, 위치 정보가 충분히 구체적이지 않았다.
- Fortune 정보는 질문 의도와 직접 관련이 약했다.

### wiki 소스 필터링 후 개선

`minecraft.wiki` 정확한 host만 허용하고, Hypixel SkyBlock, Talk 문서, April Fools 문서를 제외했다.

최종 답변:

```text
Look for diamond ore in caves, mountain biomes, and underground at y-levels 16 and below. Deepslate diamond ore is more common in deepslate and tuff blobs.
```

문제:

- 오염 소스는 줄었지만, 본문에서 가장 중요한 `Natural generation` 문단을 제대로 가져오지 못하면 답변이 여전히 애매했다.

### passage retriever 도입 후 개선

검색 결과의 스니펫만 사용하지 않고, 실제 Minecraft Wiki 페이지를 fetch해서 본문 문단을 passage로 나눈 뒤 다시 랭킹했다.

최종 답변:

```text
Look for diamond ore in caves or by mining at y-levels -58 to -64 in 1.18+ or y-levels 5-12 in older versions. Use an iron pickaxe to mine and fortune III boosts yields
```

좋아진 점:

- 버전별 Y레벨 정보가 들어갔다.
- 실제 위치 중심 답변에 가까워졌다.

남은 문제:

- `fortune III boosts yields`는 사실일 수 있지만, "어디서 찾는가"라는 질문 의도에는 부가적인 정보다.
- 도구, 인챈트, 수확량 팁은 사용자가 직접 물었을 때만 나오는 편이 좋다.

## 최신 retriever 개선

### Passage chunk overlap

커밋:

```text
54cfae3 feat: rag passage chunk 개선
```

변경 내용:

- 긴 wiki 본문을 문장 단위로 자른다.
- 다음 chunk에 이전 chunk의 마지막 문장 일부를 겹쳐 넣는다.
- `PASSAGE_MAX_CHARS`, `PASSAGE_OVERLAP_SENTENCES` 환경변수로 조절할 수 있다.

기대 효과:

- `Natural generation`처럼 앞 문맥이 중요한 문단이 잘릴 위험을 줄인다.
- passage 하나만 봐도 답변에 필요한 문맥이 더 잘 남는다.

### RRF rerank

커밋:

```text
5221bd8 perf: rag rrf rerank 성능 개선
```

변경 내용:

- keyword 순위, source 순위, semantic 순위를 RRF로 결합한다.
- 임베딩 모델이 설정되어 있으면 semantic 순위도 결합한다.
- 임베딩 모델이 없으면 keyword 순위와 source 순위만으로 RRF를 계산한다.
- `RRF_K` 환경변수로 RRF 민감도를 조절할 수 있다.
- `retrieved_passages`에 디버깅용 ranking 정보를 추가했다.

추가된 디버깅 필드:

```json
{
  "section": "Natural generation",
  "source_rank": 1,
  "keyword_rank": 2,
  "semantic_rank": null,
  "rrf_score": 0.0325
}
```

기대 효과:

- keyword 점수 하나가 최종 순위를 과하게 지배하는 문제를 줄인다.
- Tavily 검색 순위가 좋은 source와 질문 키워드에 잘 맞는 passage를 균형 있게 선택한다.
- semantic rerank가 켜졌을 때도 점수 스케일 차이 때문에 순위가 흔들리는 문제를 줄인다.

### 초보 친화 프롬프팅

커밋:

```text
b46874d chore: 초보 친화적으로 프롬포팅
```

변경 내용:

- 답변 톤을 초보 Minecraft Java Edition survival 플레이어 기준으로 조정했다.
- 직접 의도를 설명하는 데 필요한 내용은 쉽게 풀어 쓰되, Minecraft 고유 용어는 그대로 유지한다.
- 답변 길이를 보통 2-5개의 짧은 문장으로 제한한다.
- 제작법 질문이 3x3 crafting table을 요구하면 슬롯 번호로 설명한다.
- 슬롯 레이아웃은 `1 2 3 / 4 5 6 / 7 8 9`를 사용하고, 채워야 하는 슬롯과 빈 슬롯을 구분한다.
- 위치 질문은 위치, 조건, 찾는 방법에 집중한다.
- 획득이나 채굴 질문은 직접 필요한 경우에만 도구나 드롭 관련 정보를 포함한다.

기대 효과:

- 초보자가 바로 따라 할 수 있는 답변을 만든다.
- 조합법 답변에서 재료 배치를 더 명확히 전달한다.
- 관련은 있지만 질문 의도 밖인 고급 최적화나 부가 팁이 섞이는 문제를 줄인다.

## 현재 최종 답변 목표

`How do I find diamonds ?` 같은 위치 질문에는 다음처럼 짧고 목적에 맞는 답변을 목표로 한다.

최종 답변:

```text
Look for diamond ore in the Overworld between Y=14 and Y=-63 in Minecraft 1.18+, with lower levels being more common. Mining around Y=-58 is usually a good target, and caves or aquifer walls can expose diamond ore.
```

이 답변에서는 일부러 다음 내용을 제외한다.

- Fortune III
- Silk Touch
- pickaxe 추천
- 수확량 증가 팁
- 장비 최적화
- 질문과 직접 관련 없는 일반 전략

`How do I craft an iron pickaxe?` 같은 제작법 질문에는 다음처럼 슬롯 기반으로 답변하는 것을 목표로 한다.

최종 답변:

```text
Use a 3x3 crafting table. Put iron ingots in slots 1, 2, and 3, and sticks in slots 5 and 8. Leave the other slots empty.
```

## 현재 구조

현재 `rag_server.py`의 주요 흐름은 다음과 같다.

1. 사용자 질문 수신
2. SGLang으로 Minecraft Wiki 검색용 질의 재작성
3. Tavily로 Minecraft Wiki 후보 URL 검색
4. Java Edition 바닐라 야생과 맞지 않는 결과 제거
5. 검색 결과 URL의 실제 wiki 본문 fetch
6. 본문을 passage 단위로 분할
7. 긴 passage는 문장 단위로 자르고 overlap을 적용
8. 키워드 점수로 passage 후보 랭킹
9. 임베딩 모델이 설정된 경우 semantic similarity 계산
10. keyword, source, semantic 순위를 RRF로 결합
11. 선택된 passage로 context 구성
12. SGLang으로 초안 답변 생성
13. SGLang으로 근거 검증 및 off-intent detail 제거
14. 최종 `answer` 반환

답변 생성/검증 단계는 현재 다음 규칙도 함께 적용한다.

- 초보자에게 친절한 2-5문장 답변을 우선한다.
- 직접 질문한 범위 밖의 장비 최적화, 인챈트, 드롭률, 전략 팁은 제외한다.
- 3x3 제작법은 슬롯 번호로 설명한다.

## 다음 개선 후보

- 답변 언어 변환은 생성/검증 프롬프트에 섞기보다, 검증된 최종 답변을 별도 번역 단계에서 처리한다.
- `retrieved_passages`를 테스트 로그로 남겨 어떤 passage가 최종 답변에 영향을 줬는지 비교한다.
- 다이아몬드 외 질문으로도 retriever 품질을 확인한다.
  - 조합법
  - 몹 드롭
  - 바이옴 위치
  - 구조물 전리품
  - 주민 거래
  - 인챈트 메커니즘

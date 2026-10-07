# 방식·모델 선택 기록

이 문서는 Cking AI Quiz Pilot에서 영상 분석 방법을 정한 근거, Pilot v1 실행 기록, 현재 Pilot v2의 실행 프로토콜과 결과 해석 기준, Pilot v2-r2 결과와 그에 따른 Quiz 생성 방식 결정, 이후의 production architecture 변경을 기록합니다. Pilot v1(2~3장)과 Pilot v2(6~8장)는 별개의 실험입니다. 근거의 성격은 다음 표기로 구분합니다.

- **USER DECISION**: 사람이 정한 요구사항이나 결정입니다. 저장소 코드로 증명된 사실이 아닙니다.
- **CONFIRMED BY OFFICIAL DOCS**: 2026-10-02에 각 Provider 공식 문서에서 확인한 사실입니다. 문서는 이후 바뀔 수 있습니다.
- **OBSERVED IN PILOT**: 이 저장소의 Pilot 실행 결과에서 관찰한 사실입니다.
- **CURRENT PILOT POLICY**: 현재 Pilot에 적용하는 실행 규칙입니다.

## 1. 영상 분석 방법: Gemini Video Grounding 유지

### Cking 요구사항 (USER DECISION)

크리에이터가 YouTube URL을 입력하면 영상을 분석하고, 그 분석 결과로 Quiz를 생성합니다. 이는 서비스 요구사항으로 정한 사항이며 저장소에서 증명된 사실이 아닙니다. 2026-10 기준 Cking-BE AI Quiz는 `contentText`를 입력받는 계약이고, YouTube 영상 분석 pipeline은 없습니다.

### 선정 근거

Gemini Video Grounding을 현재 영상 분석 방법으로 유지합니다. 근거는 **현재 Cking 입력 계약인 YouTube URL 직접 입력과 가장 직접적으로 연결된다**는 점입니다.

- **CONFIRMED BY OFFICIAL DOCS**: Gemini API는 공개 YouTube URL을 video input으로 직접 받습니다. 그래서 영상 파일을 따로 확보하거나 업로드하는 pipeline 없이 연결할 수 있습니다.
- 저장소 코드: 이 저장소의 Gemini adapter는 이미 YouTube URL을 video input으로 전달합니다(`src/provider_adapters.py`).
- **이 선정은 품질 비교의 결과가 아닙니다.** Gemini, Azure Content Understanding, TwelveLabs Pegasus의 영상 분석 품질을 같은 조건에서 실측 비교하지 않았습니다. 따라서 Gemini의 분석 품질이 다른 Provider보다 낫다고 주장하지 않습니다.

알려진 제한은 다음과 같습니다.

- **CONFIRMED BY OFFICIAL DOCS**: YouTube URL 입력은 Preview 기능입니다. 공개 영상만 가능하며 일부공개·비공개 영상은 사용할 수 없습니다. 무료 tier는 하루 8시간 한도가 있습니다.
- **CONFIRMED BY OFFICIAL DOCS**: Gemini가 YouTube 자막(caption)을 사용하는지, agentic 처리 모드가 불러오는 transcript가 어디서 오는지 등 내부 처리 방식은 공식 문서에 공개되어 있지 않습니다. 영상 분석 결과를 해석할 때 이 한계를 고려합니다.
- **OBSERVED IN PILOT**: 첫 영상에서 Grounding은 attempt 1~4가 실패한 뒤 attempt 5에 성공했고, Direct Quiz는 HTTP 503으로 3회 모두 완료되지 않았습니다. 운영 안정성을 보장하는 근거는 아직 없습니다.

### 다른 후보 (CONFIRMED BY OFFICIAL DOCS)

두 후보는 근거 수준이 다르므로 구분해 기록합니다.

| 후보 | 공식 문서에서 확인한 입력 방식 | YouTube URL 직접 입력 |
| --- | --- | --- |
| Azure Content Understanding | 직접 업로드(`analyzeBinary`, 200MB·30분 이하), storage URL 참조(`analyze`, 4GB·2시간 이하) | 공식 문서에서 **확인되지 않음** (언급 없음) |
| TwelveLabs Pegasus | asset, 원본 미디어 파일로 바로 연결되는 URL, base64 | **미지원이 명시됨** ("Video hosting platforms … are not supported") |

따라서 현재 Cking 요구사항에서 두 Provider를 쓰려면 영상 파일 확보, 임시 저장, Provider 업로드, 권리 확인, 삭제 정책, 실패 처리, 경우에 따라 비동기 polling까지 추가 pipeline이 필요합니다. 이번 프로젝트에서는 이 pipeline을 만들지 않습니다.

### 3사 Pre-study: DEFER (USER DECISION)

세 Provider를 같은 영상으로 실측 비교하는 Pre-study는 현재 범위에서 실행하지 않습니다. 다음과 같은 경우 조사 결과를 재검토 자료로 사용합니다.

- 서비스 요구사항이 바뀌는 경우
- 영상 파일 업로드를 허용하게 되는 경우
- Gemini를 대체할 Provider가 필요해지는 경우

아래 표는 2026-10-02 공식 문서 조사 기준의 요약입니다(CONFIRMED BY OFFICIAL DOCS). 문서는 바뀔 수 있으므로 재검토할 때는 공식 문서를 다시 확인해야 합니다. Gemini는 1장 "선정 근거"에서 따로 다루므로 표에 포함하지 않았습니다.

| 항목 (2026-10-02 조사 기준) | Azure Content Understanding | TwelveLabs Pegasus |
| --- | --- | --- |
| 처리 방식 | 비동기(Analyze 후 결과 조회). 결과는 최대 24시간 보관 | 사전 인덱싱 없이 analyze 가능. 1시간 미만은 동기, 최대 2시간은 비동기 task |
| 구조화 출력 | `fieldSchema`의 generate/classify field. 생성 모델(Azure OpenAI 계열) 배포 연결 필요 | `response_format`으로 JSON schema 지정 가능. strict 준수 보장은 확인하지 못함 |
| 언어 | 한국어 음성 전사 지원(Azure Speech fast transcription `ko-KR`). 생성 출력 언어의 공식 목록은 없음 | 한국어는 partial support |
| 과금 단위 | 영상 추출 시간당 과금 + context processing token + 연결한 생성 모델 token 별도 | 입력 영상 시간당 과금 + 출력 token |
| 데이터 정책 | Content Understanding 단계의 입력·중간 데이터는 처리 후 삭제, 결과는 최대 24시간 보관. 연결된 Azure OpenAI 처리에는 Azure OpenAI 데이터 정책이 별도로 적용되며, 그 학습 사용 여부는 이번 조사에서 확인하지 않음 | 계정 설정에서 opt-out하지 않으면 고객 데이터를 학습에 사용 |
| 권리 | 제출 콘텐츠에 필요한 권리 확보를 이용자에게 요구 | 업로드할 권리 보장을 이용자에게 요구 |

## 2. Pilot v1 실행 프로토콜 (pilot-v1 기록)

이 장은 `pilot-v1`(`configs/pilot.yaml`) repetition 1을 실행할 때 적용한 프로토콜의 기록입니다. 기존 Pilot v1 실행과 결과의 해석 근거로 그대로 유지하며, Pilot v2에는 적용하지 않습니다. Pilot v2 프로토콜은 6장을 따릅니다.

### 범위

- 현재 Pilot 단계에서는 4개 영상의 repetition 1을 완료합니다.
- 첫 영상 `nasa-water-cycle-2019`는 기존 결과를 유지합니다(3장 참고).
- 남은 3개 영상 `nasa-methane-2020`, `nasa-mars-organics-2025`, `kari-microgravity-2024`에 이 프로토콜을 적용합니다.
- repetition 2는 현재 범위에서 실행하지 않습니다. 고정 입력 manifest(영상·`promptVersion`당 `contentText` 하나)와 aggregator(Grounding·Quiz의 repetition 일치 요구) 계약 때문에, repetition 1 완료 후 의미와 계약을 별도로 결정합니다. 이 결정은 Pilot v2에서 내렸으며 6장에 기록합니다.

### 기준과 조건

- 코드 기준 commit: `a534b83cb9d13a2d43676f1d276f36cf077c0b00` (main, PR #11 Direct `validatorStatus` 보존과 PR #13 Human Evaluation 검증 반영). 남은 3개 영상의 Pilot 실행 코드는 이 commit을 기준으로 합니다. 첫 영상은 이 commit 이전에 실행했으며 그 실행 맥락은 3장에 따로 기록합니다. 이 문서를 반영한 이후 main HEAD는 앞으로 이동합니다. 그 사이 commit이 문서만 바꾼 것이라면, Pilot 실행 코드가 `a534b83`과 같은지 확인한 뒤 해당 main HEAD에서 실행할 수 있습니다. 코드가 바뀐 commit에서는 같은 조건으로 간주하지 않습니다.
- 결과 위치: 기본 `results/`. 테스트·fixture·dry-run 성격의 실행은 이 기본 위치에 만들지 않습니다(아래 "fixture와 scratch 결과" 참고).
- 조건과 모델

| 조건 | 방식 | 모델 |
| --- | --- | --- |
| Grounding | `gemini_video` | `gemini-3.8-flash` |
| Quiz A | 승인된 고정 `contentText` | `gemini-3.8-flash` |
| Quiz B | Quiz A와 같은 승인 `contentText` | `gpt-5.4-mini` |
| Direct C | `gemini_direct_quiz` | `gemini-3.8-flash` |

- Quiz는 `promptVersion=pilot-v1`, 영상당 3문항, 4지선다입니다.

### CLI 값

| 옵션 | 값 | 적용 조건 |
| --- | --- | --- |
| `--video-processing` | `static` | Grounding, Direct C |
| media resolution | 설정하지 않음 (API 기본값) | Grounding, Direct C |
| Gemini thinking | 설정하지 않음 (모델 기본값) | Gemini 호출 |
| `--max-output-tokens` | `8192` | 모든 조건 |
| `--timeout-seconds` | `120` | 모든 조건 |
| `--retry-attempts` | `1` | 모든 조건 |
| `--call-limit` | `2` | 모든 조건 |
| `--per-call-cost-limit` | `0.10` (USD) | 모든 조건 |
| `--total-cost-limit` | `0.20` (USD) | 모든 조건 |
| `--video-estimated-input-tokens` | `nasa-methane-2020`: `16000`<br>`nasa-mars-organics-2025`: `13000`<br>`kari-microgravity-2024`: `31000` | Grounding, Direct C |
| `--quiz-estimated-input-tokens` | `2000` | Quiz A, Quiz B |
| `--reasoning-effort` | `low` | Quiz B (OpenAI) |

- `--reasoning-effort low`는 남은 3개 영상의 repetition 1부터 적용하는 이 프로토콜의 결정값입니다. 첫 영상에서 사용한 값을 복원한 것이 아니며, 첫 영상과 같은 reasoning 조건이었다고 주장하지 않습니다.
- 비용 한도는 CLI 실행 한 번에만 적용되는 추정 비용 guard입니다. 실제 청구액 상한이나 여러 실행에 걸친 누적 상한이 아닙니다.

### technical attempt와 HTTP retry

두 가지 재시도는 단위가 다릅니다.

- **technical attempt**: CLI 실행 한 번이며 결과 행 하나로 기록됩니다(`attempt` 번호). Grounding, Quiz A, Quiz B, Direct C 모두 조건당 최대 3회입니다.
- **HTTP retry**: 한 technical attempt 안에서 HTTP 429/5xx 응답 후 즉시 다시 보내는 요청입니다. `--retry-attempts 1`, `--call-limit 2`이므로 technical attempt 하나당 HTTP 요청은 최대 2회입니다. timeout, 네트워크 오류, 일반 4xx는 HTTP retry를 하지 않습니다.

technical attempt로 계산하는 기준은 "CLI 실행으로 benchmark 결과 행이 실제로 생성됨"입니다.

- live guard 거부로 `errorCategory=live_guard` 결과 행이 생성되면, Provider HTTP 요청이 0회였더라도 technical attempt를 소모한 것으로 봅니다.
- `parser.error`로 끝난 실행처럼 결과 행 자체가 생성되지 않은 실행은 technical attempt가 아닙니다. 승인·SHA-256·고정 입력 manifest 검사에서 거부된 Quiz 실행도 결과 행이 생성되기 전에 끝나므로 technical attempt가 아닙니다.
- 운영자 입력 실수로 attempt를 소모하지 않도록, Provider 호출 전에 실제 명령을 이 문서의 고정 CLI 값과 대조합니다.

미완료 처리 규칙은 다음과 같습니다.

- 3 technical attempts 후에도 완료되지 않은 조건은 "3 technical attempts incomplete"로 기록합니다. 이는 품질 실패가 아닙니다.
- Grounding이 미완료되면 승인할 source가 없으므로, 그 영상의 Quiz A·B는 실행하지 못한 것으로 기록합니다.

### 조건 완료와 재실행

- 각 조건(Grounding, Quiz A, Quiz B, Direct C)은 처음으로 `apiStatus=success`가 기록된 technical attempt에서 완료되며, 성공한 조건은 다시 실행하지 않습니다.
- `apiStatus=error`인 경우에만 다음 technical attempt를 실행할 수 있고, 3회 모두 `apiStatus=error`이면 "3 technical attempts incomplete"로 기록하고 중단합니다.
- Quiz A·B·Direct C에서 Provider 응답을 정상적으로 받아 `apiStatus=success`가 된 뒤의 `parseStatus=fail`, `validatorStatus=fail`, `errorCategory=quiz_contract_error`·`evidence_not_in_content` 등은 모델 출력의 품질·형식 결과입니다. technical failure가 아니므로 재실행하지 않으며, validator가 통과할 때까지 다시 생성하지 않습니다.
- Grounding은 다음 단계에 쓸 `contentText`가 필요하므로, 구조적으로 사용할 수 없는 Provider 응답은 실행기가 `apiStatus=error`(`errorCategory=invalid_grounding_response`)로 기록하며 재시도 대상입니다. Quiz와의 이 비대칭은 의도된 정책입니다. Grounding도 `apiStatus=success` 이후에는 사람 평가 결과와 관계없이 재실행하지 않습니다.

### 결과 상태 (코드 기준 commit 기준)

Direct C는 고정 `contentText`가 없어 evidence 포함 검사를 하지 않으며 `beCompatibility=not_applicable`, `evidenceTextContained=null`입니다. 공통 Quiz 구조 검사 결과는 다음과 같이 기록됩니다.

| Direct C 응답 | `apiStatus` | `parseStatus` | `validatorStatus` | `errorCategory` |
| --- | --- | --- | --- | --- |
| parse 통과 + 공통 구조 통과 | `success` | `pass` | `not_run` | `null` |
| parse 통과 + 공통 구조 실패 | `success` | `pass` | `fail` | `quiz_contract_error` |
| parse 실패 | `success` | `fail` | `not_run` | `parse_error` |

구조 실패도 `validatorStatus=not_run`으로 덮어쓰던 이전 동작은 PR #11에서 수정됐습니다.

Human Evaluation 판정 값은 `pass`, `fail`, `uncertain`, `null`만 허용합니다(PR #13). 이 검사는 저장된 값의 유효성 검사이며 평가 완료 여부 검사가 아닙니다. 미평가 `null`과 `fail`은 정상 값입니다. 전체 JSON Schema 검증이 아니라 Human Evaluation 계약(판정 값, `reviewNote`, `questionIndex`, `groundingFacts`·`questionReviews` 배열과 항목 key)만 검사합니다. 세부 계약은 [결과 형식](results-format.md)을 따릅니다.

- 새 실행·approval 전 storage integrity 검사: summary 파일의 모든 행을 검사하며, 잘못된 Human Evaluation 행이 하나라도 있으면 실행·승인을 중단합니다.
- offline 집계: 선택한 Grounding·Quiz 행만 검사합니다. 선택하지 않은 과거 행의 잘못된 값은 집계를 막지 않습니다.

### Human Evaluation 수동 기록 절차

Human Evaluation은 summary JSONL에 사람이 직접 기록합니다. 실행기는 동시 writer를 지원하지 않으므로, 다음은 Pilot 운영 절차상의 safeguard입니다.

1. 해당 benchmark 실행이 진행 중이지 않은지 확인합니다. CLI가 같은 summary를 다시 쓰는 동안에는 편집하지 않습니다.
2. 편집 직전에 summary JSONL의 최신 내용을 다시 읽습니다.
3. 대상 `runId`를 다시 확인합니다.
4. 그 행의 Human Evaluation 필드만 수정합니다.
5. 저장합니다.
6. 저장 후 JSON 형식과 Human Evaluation 계약(허용값)을 다시 확인합니다.
7. 이 확인이 끝나기 전에는 다음 실행이나 approval을 진행하지 않습니다.

### fixture와 scratch 결과

- 테스트·fixture·dry-run 성격의 결과는 기본 `results/`에 만들지 않고, 별도의 scratch `--results-dir`(실행기가 허용하는 `results/` 하위 경로)을 사용합니다. 실제 Pilot 결과와 같은 summary 파일을 공유하지 않기 위해서입니다.
- 단, 고정 입력 manifest는 `--results-dir`과 무관하게 `data/restricted/fixed-content/`를 공유합니다. 따라서 실제 Pilot과 같은 `videoId` + `promptVersion`을 사용하는 Quiz fixture·검증 실행은 Pilot 완료 전에 수행하지 않습니다. scratch `--results-dir`만으로 고정 입력 manifest가 격리된다고 간주하지 않습니다.

### 영상별 실행 순서

1. Grounding
2. Grounding Human Evaluation: fact별 `factExists`·`evidenceTypeCorrect`·`timestampAccurate`, 실행 단위 `omission`·`hallucination`
3. `approve_content(runId, approved_by="taeyeonon")`
4. 승인된 Grounding의 `results/evaluation/<runId>.json`에 있는 `contentText`를 내용 변경 없이 Git-ignore된 로컬 content file(`data/restricted/` 아래)로 준비
5. 첫 Quiz 실행 직전 고정 입력 재확인(아래 목록)
6. 같은 `--content-file`과 `--source-grounding-run-id`(승인한 Grounding의 `runId`)를 사용해 Quiz A, Quiz B 실행
7. Quiz A·B 질문별 Human Evaluation
8. Direct C
9. Direct C 질문별 Human Evaluation

`approved_by`의 `taeyeonon`은 사람이 제공한 approval tracking identifier이며 인증 정보가 아닙니다.

content file은 승인된 `contentText`와 정확히 같아야 합니다. 요약, 수정, 재작성은 물론 끝 줄바꿈을 포함해 어떤 문자도 추가하거나 빼지 않습니다. 현재 standalone Quiz는 Provider 호출 전에 다음을 검사하고, 하나라도 맞지 않으면 실행을 거부합니다.

- `--source-grounding-run-id`가 같은 결과 집합의 성공한 Grounding 행 하나를 가리키고, 그 행이 같은 영상이며 `contentTextApprovalStatus=approved`이고 승인 추적 정보가 유효함
- 그 행의 evaluation `contentText`의 UTF-8 SHA-256이 행에 기록된 승인 해시(`contentTextSha256`)와 같음
- `--content-file` 텍스트의 UTF-8 SHA-256이 같은 승인 해시와 같음
- 고정 입력 manifest: 영상과 `promptVersion`마다 처음 사용한 `contentText` 해시가 기록되며, 이후 다른 해시로는 실행할 수 없음. 그래서 Quiz A와 B는 같은 content file을 써야 함

manifest는 그 영상의 첫 Quiz Provider 호출 전에 입력을 고정합니다. 그래서 각 영상의 Grounding Human Evaluation과 approval이 끝난 뒤, 첫 Quiz(A 또는 B)를 실행하기 직전에 다음을 사람이 다시 확인합니다.

- `videoId`
- 승인한 Grounding `runId`와 명령의 `--source-grounding-run-id`가 같음
- 그 Grounding 행이 `contentTextApprovalStatus=approved`이고 `approvedBy`·`approvedAt`이 기록됨
- evaluation `contentText`의 SHA-256이 Grounding 행의 `contentTextSha256`과 같음
- 명령의 `--content-file`이 의도한 파일이고, 그 내용의 SHA-256이 같은 승인 해시와 같음
- `promptVersion`이 `pilot-v1`

source `runId`나 content file이 잘못됐다면 Provider 호출 전에 바로잡고 다시 확인합니다. 첫 Quiz 실행 이후에는 A/B 결과를 보고 `contentText`를 바꾸지 않으며, A와 B는 반드시 같은 승인 `contentText`를 사용합니다.

### transcript 조건

Transcript condition was not executed in this Pilot. 권한 있는 transcript를 입력으로 쓰는 조건은 이번 Pilot에서 실행하지 않으며, `not_run` 결과 행도 만들지 않습니다.

### 가격 출처

- Gemini `gemini-3.8-flash`: 입력 $0.75 / 출력(thinking 포함) $3.75 per 1M token. 출처는 [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)이며 2026-10-02에 확인했습니다. 공식 페이지 기준 이 단가는 2026-12-31까지 적용됩니다.
- OpenAI `gpt-5.4-mini`: 이 문서에서 단가를 정하지 않습니다. Quiz B를 실제로 호출하기 직전에 공식 OpenAI 가격을 다시 확인하고, 그 시점의 단가와 확인 날짜를 실행 인자(`--pricing-reference` 포함)로 사용합니다.

### 실행 기록

각 실행의 실제 명령(API key 등 비밀값 제외)과 실행 시각을 남깁니다.

## 3. 첫 영상: protocol 이전 실행 (OBSERVED IN PILOT)

`nasa-water-cycle-2019`의 repetition 1은 이 프로토콜을 정하기 전에 실행했습니다. 실제 CLI 값과 실행 당시의 정확한 working tree(프롬프트를 바꾼 commit `b1989e5` 이전 상태)를 복원할 수 없습니다. 따라서 남은 3개 영상과 완전히 같은 실행 조건이었다고 주장하지 않습니다.

- **Grounding**: attempt 1~4가 실패한 뒤 attempt 5가 성공했습니다. 사람 평가를 마쳤고 `contentText`를 승인했습니다(승인 추적 필드가 없는 기존 승인). protocol 이전 실행이므로 현재 상한(3회)을 넘은 기록이어도 그대로 유지합니다.
- **Quiz A/B**: 같은 승인 `contentText`를 사용했으므로 첫 영상 안의 A/B 비교 결과는 유지합니다. 두 Quiz의 실행과 사람 평가가 완료돼 offline 집계가 가능합니다.
- **Direct C**: 3 technical attempts incomplete due to HTTP 503. 세 번째 시도에는 `providerErrorCode=service_unavailable`이 기록됐습니다. 품질 실패가 아니며 추가 attempt는 하지 않습니다.
- 기존 결과는 수정하거나 backfill하지 않습니다. 이후 반영된 PR #11(Direct `validatorStatus` 보존)과 PR #13(Human Evaluation 검증) 때문에 첫 영상 결과를 다시 쓰지 않습니다.

## 4. 선택 근거 기록 원칙

결과를 선택 근거로 사용할 때 Video Grounding, 고정 입력 Quiz Generation, End-to-End 결과를 구분해 기록합니다. 각 선택에는 사용 데이터와 권리 확인, 실행 날짜, 모델 ID, 프롬프트 버전, 가격 출처, 자동 지표, 실제 영상과 승인된 ground truth를 대조한 사람 검수 근거, 제한사항 및 Cking-BE 적용 결정을 남깁니다. 직접 Quiz의 `beCompatibility`는 `not_applicable`입니다. Pilot v2-r2 결과와 최종 Quiz 생성 방식 선정은 8장에 기록합니다.

## 5. 결과 해석 caveat와 후속 과제

다음은 현재 Pilot의 blocker가 아닙니다.

Pilot v1 결과 해석 시 함께 기록할 caveat:

- Quiz A와 Direct C의 차이는 분리 방식과 직접 방식의 차이만이 아닙니다. 입력(승인된 텍스트와 영상), 사람 검수·승인 단계, evidence 포함 검사 적용 여부도 다릅니다.
- Quiz A와 B는 같은 승인 `contentText`를 쓰지만, Gemini thinking 기본값과 OpenAI `reasoning-effort low`, Provider별 구조화 출력 schema 적용 방식이 다릅니다. 이 reasoning 설정 비대칭은 Pilot v1의 한계로 남습니다. Pilot v2의 caveat는 7장에 따로 기록합니다.

별도 코드 후속 과제(이번 Pilot 프로토콜에 포함하지 않음). 6장의 Pilot v2 기준 commit에서 다시 확인한 상태입니다.

- offline aggregator가 선택된 Quiz의 `parseStatus`·`validatorStatus`·`errorCategory` 등을 충분히 노출하지 않음: 남아 있음
- timestamp 시작·끝과 영상 길이 범위 검증의 제한: 시작·끝 정합성 검사는 이후 반영됐습니다(PR #18). 영상 길이 상한 검사는 처음에 `kari-microgravity-2024`에만 적용됐지만, Issue #31에서 NASA 세 편에도 `durationSeconds`를 기록해 이제 네 영상 모두에 적용됩니다. 다만 길이 검사는 영상 길이를 넘는 값만 잡으므로, MM:SS형 숫자 문제는 Grounding `video-grounding-v3`의 문자열 timestamp 변환으로 막습니다(6장 "첫 실행 중단과 재시작")
- 중복 `runId`의 전역 탐지 제한: 남아 있음
- NaN·Infinity 처리 계약의 경로별 차이: 남아 있음
- 전역 storage preflight의 승인 추적 검증 범위 제한: Pilot v2 Grounding 검수 상태의 일관성 검사는 추가됐지만, 범위 제한은 일부 남아 있음
- 실행 설정 전체 snapshot 부재: 남아 있음. 그래서 6장의 실행 기록에 실제 명령을 남깁니다.

## 6. Pilot v2 실행 프로토콜 (CURRENT PILOT POLICY)

### 범위

- Pilot v2는 pilot-v1과 별개의 실험입니다. 설정은 `configs/pilot-v2.yaml`이며 `src.pilot_runner`의 모든 Grounding·Quiz·Direct 생성 명령에 `--config configs/pilot-v2.yaml`을 붙입니다. Human blind evaluation과 Judge 같은 평가 도구는 각 도구의 명령과 설정 계약을 따르며, 이 생성 설정을 `--config`로 넘기지 않습니다. 기존 Pilot v1 결과·고정 입력 manifest·설정(`configs/pilot.yaml`)은 바꾸지 않고, Pilot v1 결과를 Pilot v2 집계나 평가에 포함하지 않습니다.
- 영상 4개(`nasa-water-cycle-2019`, `nasa-methane-2020`, `nasa-mars-organics-2025`, `kari-microgravity-2024`) × repetition 2(`repetitions_per_condition: 2`) × 조건 A/B/C입니다.
- transcript 조건은 Pilot v2 범위에 포함하지 않으며 실행하지 않습니다.

### 기준 commit

두 commit을 구분해 기록합니다.

- **프로토콜·코드 기준 commit**: 처음에는 `6bedc74706a7d4cbc5d2544a450365c20d6fb330` (main, PR #29 반영)으로 이 문서의 Pilot v2 계약을 확인했습니다. Pilot v2 Grounding 검수(PR #22), Quiz·Direct 생성 조건(PR #24), Human blind evaluation(PR #26), repetition별 고정 입력(PR #29)을 포함합니다. 첫 실행이 timestamp 문제로 중단된 뒤(아래 "첫 실행 중단과 재시작"), 재시작의 기준 commit은 Issue #31 수정(Grounding `video-grounding-v3`, NASA `durationSeconds`)이 main에 반영된 commit입니다. 그 SHA는 재시작 전에 확인해 실행 기록에 남깁니다.
- **실제 실행 commit**: 실제 Pilot v2를 실행하기 직전에 checkout한 HEAD입니다. 실행 기록에 따로 남깁니다. 기준 commit과 다르면 그 사이에 Pilot 실행 코드가 바뀌지 않았는지 확인합니다(예: `git diff --stat <기준 commit> HEAD -- src configs data/videos.jsonl`). 실행 코드가 바뀐 commit에서는 이 프로토콜과 같은 조건으로 간주하지 않고 사람이 별도로 결정합니다.

### 첫 실행 중단과 재시작 (OBSERVED IN PILOT)

- 2026-10-07 main `df71ba7`에서 `results/pilot-v2`로 첫 Pilot v2 조건인 `nasa-water-cycle-2019` repetition 1 Grounding을 실행했습니다(runId `1a77bc67d5724f77bfbd6ee2ef5af0d3`, Grounding `video-grounding-v2`, `apiStatus=success`).
- 사람이 원본 영상과 비교한 결과, facts timestamp가 경과 초가 아니라 MM:SS의 콜론만 뺀 숫자였습니다(예: 01:41.2 → 141.2, 02:26.7 → 226.7, 03:04.5 → 304.5). prompt의 경과 초 지시를 Provider가 지키지 않았고, 숫자 하나만으로는 이 오류를 판별할 수 없으며, 당시 이 영상에는 `durationSeconds`가 없어 길이 검사도 적용되지 않았습니다(Issue #31).
- Pilot v2는 이 시점에 중단했습니다. 이 Grounding은 검수하지 않았고, 이후 Quiz A/B와 Direct C, 다른 영상은 실행하지 않았습니다.
- `results/pilot-v2`는 실제 Provider 출력이자 이 문제의 근거로 그대로 보존합니다. 수정·삭제·이동하거나 timestamp를 사후 보정하지 않습니다. 이 결과는 Pilot v2 집계·평가에 포함하지 않으며, 수정 후 Pilot v2 Quiz는 `video-grounding-v2` Grounding을 원천으로 받지 않습니다.
- 수정 후 Pilot v2는 Grounding `video-grounding-v3`로 `results/pilot-v2-r2`에서 첫 영상·repetition부터 다시 시작합니다. `video-grounding-v3`는 Provider에게 timestamp를 `"MM:SS"` 문자열로 받고 실행기가 경과 초로 변환해 저장합니다. 길이 검사를 위해 NASA 세 편에도 `durationSeconds`를 기록했지만, 이는 영상 길이를 넘는 값만 잡는 추가 방어입니다. 재시작 실행 결과는 8장에 기록합니다.

### 조건과 모델

| 조건 | 실행 | 모델 | 생성 설정 |
| --- | --- | --- | --- |
| Grounding | `grounding --method gemini_video`, Grounding `promptVersion=video-grounding-v3` | `gemini-3.8-flash` | thinking 설정하지 않음 (모델 기본값) |
| Quiz A | `quiz`, 같은 repetition의 승인 `contentText` | `gemini-3.8-flash` | `thinking_level: medium` (config) |
| Quiz B | `quiz`, Quiz A와 같은 repetition의 같은 승인 `contentText` | `gpt-5.4-mini` | `reasoning_effort: medium` (config) |
| Direct C | `end-to-end --method gemini_direct_quiz`, YouTube 영상 직접 입력 | `gemini-3.8-flash` | `thinking_level: medium` (config) |

- Quiz A/B와 Direct C는 `promptVersion=pilot-v2`이며, prompt로 정확히 3문항·문항당 보기 4개를 요청합니다. Quiz A/B 입력은 승인된 `contentText`뿐이며 facts는 넣지 않습니다.
- 구조화 출력: Gemini는 JSON schema 응답 형식, OpenAI는 strict JSON schema를 사용합니다. schema는 필드 구조만 정하고 문항 수·보기 수(배열 길이)는 강제하지 않습니다. 따라서 3문항은 Provider가 보장하는 값이 아니라 결과로 측정하는 값입니다.
- temperature·top_p는 설정하지 않습니다.
- Quiz A/B·Direct C의 thinking/reasoning 값은 config에서 가져옵니다. 실행기는 config가 위 모델·`medium` 설정·3문항·보기 4개·`retry_attempts: 0`과 다르면 실행을 거부합니다. CLI의 `--thinking-level`·`--reasoning-effort`는 Pilot v2에서 사용하지 않습니다. config와 다른 값을 넘기면 Provider 호출 전에 거부됩니다.
- Gemini의 `medium`과 OpenAI의 `medium`은 각 Provider가 정한 단계 이름입니다. 두 설정이 같은 추론량이나 같은 계산량을 뜻한다고 간주하지 않습니다.

### CLI 값

| 옵션 | 값 | 적용 조건 |
| --- | --- | --- |
| `--config` | `configs/pilot-v2.yaml` | 모든 조건 |
| `--results-dir` | `results/pilot-v2-r2` | 모든 조건 (아래 "결과 디렉터리" 참고) |
| `--video-processing` | `static` | Grounding, Direct C |
| media resolution | 설정하지 않음 (API 기본값) | Grounding, Direct C |
| `--max-output-tokens` | `8192` | 모든 조건 |
| `--timeout-seconds` | `120` | 모든 조건 |
| `--retry-attempts` | Grounding `1`, Quiz A/B·Direct C `0` | 조건별 |
| `--call-limit` | Grounding `2`, Quiz A/B·Direct C `1` | 조건별 |
| `--per-call-cost-limit` | `0.10` (USD) | 모든 조건 |
| `--total-cost-limit` | `0.20` (USD) | 모든 조건 |
| `--video-estimated-input-tokens` | `nasa-methane-2020`: `16000`<br>`nasa-mars-organics-2025`: `13000`<br>`kari-microgravity-2024`: `31000`<br>`nasa-water-cycle-2019`: `29000` | Grounding, Direct C |
| `--quiz-estimated-input-tokens` | `2000` | Quiz A, Quiz B |
| `--thinking-level`, `--reasoning-effort` | 사용하지 않음 (config 값 사용) | Quiz A/B, Direct C |
| `--input-price-per-million`, `--output-price-per-million`, `--pricing-reference` | 실행 직전에 확인한 해당 Provider 단가와 출처·확인 날짜 | 모든 조건 |

- Pilot v2 Quiz A/B·Direct C는 `--retry-attempts 0`일 때만 실행됩니다. 다른 값이나 생략은 Provider 호출 전에 거부됩니다. Grounding의 HTTP retry 정책은 Pilot v1과 같습니다.
- `--call-limit`은 CLI 실행 한 번의 HTTP 요청 수(retry 포함) 상한입니다. Grounding은 technical attempt 하나에 HTTP 요청이 최대 2회, Quiz A/B·Direct C는 1회입니다.
- 각 조건의 CLI 실행은 Provider 하나만 호출하므로 해당 Provider 단가를 일반 가격 옵션으로 넘깁니다.

### 비용 guard와 가격 재확인

- `--per-call-cost-limit`·`--total-cost-limit`은 CLI 실행 한 번(그 실행의 Provider router)에만 적용되는 추정 비용 guard입니다. 실제 청구액 상한이 아니며, Pilot v2 실험 전체나 여러 실행에 걸친 누적 비용 상한도 아닙니다.
- 실행기는 HTTP 요청마다 `입력 token 추정치 × 입력 단가 + --max-output-tokens × 출력 단가`로 비용을 추정하고, 한도를 넘으면 Provider 호출 전에 거부합니다.
- `0.10`·`0.20` USD와 입력 token 추정치는 영구히 안전한 값이 아닙니다. Provider 가격은 바뀔 수 있으므로 실제 Pilot v2 실행 직전에 Gemini와 OpenAI 공식 가격을 다시 확인합니다. 그 단가로 Grounding(HTTP 요청 최대 2회)·Quiz·Direct의 추정 비용이 `--per-call-cost-limit`과 `--total-cost-limit` 안에 드는지, 입력 token 추정치가 충분한지 확인합니다.
- 확인한 단가·출처·확인 날짜를 `--pricing-reference`와 실행 기록에 남깁니다. 값을 바꿔야 하면 실행 전에 사람이 정하고 이 문서에 기록합니다. 2장의 가격 출처는 Pilot v1 실행 당시의 기록입니다.

### 결과 디렉터리와 고정 입력 manifest

- Pilot v2의 모든 명령은 같은 `--results-dir results/pilot-v2-r2`를 사용합니다. Grounding·Quiz·Direct 실행, Grounding 검수, Human blind evaluation, Judge가 모두 이 디렉터리를 읽거나 씁니다. `results/pilot-v2`는 중단된 첫 실행의 보존 기록이므로 어떤 명령도 그 디렉터리에 쓰지 않습니다.
- Pilot v1 결과(`results/`)와 분리하는 이유는 다음과 같습니다.
  - Judge는 성공한 A/B Quiz를 `videoId`·`repetition`·모델로 찾고 `promptVersion`으로 나누지 않습니다. 같은 디렉터리에 Pilot v1·v2 성공 행이 함께 있으면 같은 조건의 성공 Quiz가 둘이 되어 Judge가 계획 단계에서 중단합니다.
  - 조건당 성공 후보 규칙, attempt 번호, storage integrity 검사는 결과 디렉터리 단위로 적용됩니다.
- 같은 결과 디렉터리에서는 Pilot 작업이 한 번에 하나만 실행됩니다. 실행기는 결과 디렉터리마다 OS 잠금(`.pilot.lock`)을 잡고, 이미 실행 중이면 기다리지 않고 바로 실패합니다. Human blind evaluation의 `create`·`import`도 같은 잠금을 사용합니다. 그래서 `results/pilot-v2-r2`에 대한 명령을 동시에 실행하지 않습니다.
- 고정 입력 manifest는 결과 디렉터리와 별개입니다. `--results-dir`과 무관하게 `data/restricted/fixed-content/`에 기록되며 모든 결과 디렉터리가 공유합니다.
  - Pilot v1: 영상·`promptVersion`마다 하나의 고정 `contentText`입니다(2장). 이 의미와 기존 항목은 바뀌지 않았습니다.
  - Pilot v2: 영상·`promptVersion`·repetition마다 하나의 고정 `contentText`입니다.
  - 기존 manifest 항목은 Pilot v1뿐이었으므로 migration은 하지 않았습니다.
- fixture·scratch·검증 실행은 `results/pilot-v2-r2`에 만들지 않고 별도의 scratch `--results-dir`(`results/` 하위)을 사용합니다. 다만 manifest는 공유되므로, 실제 Pilot v2 영상의 `videoId`로 `pilot-v2` Quiz fixture·검증 실행을 하면 그 영상·repetition의 고정 입력이 먼저 등록됩니다. 따라서 Pilot v2 완료 전에는 실제 영상 ID로 `pilot-v2` Quiz fixture·검증 실행을 하지 않습니다. 자동 테스트는 임시 저장소를 사용하므로 이 manifest를 쓰지 않습니다.

### repetition 계약

각 repetition은 Grounding부터 독립적으로 실행합니다(USER DECISION).

```
repetition r: Grounding(r) → Grounding 검수(r) → 승인 contentText(r) → Quiz A(r), Quiz B(r)
```

- 각 repetition은 자기 Grounding을 실행하고 검수합니다. 그 repetition에서 승인된 `contentText`가 그 repetition의 고정 입력이 됩니다.
- 같은 repetition의 Quiz A와 B는 같은 승인 `contentText`를 사용합니다. 같은 repetition에서 다른 `contentText`로 Quiz를 실행하려 하면 Provider 호출 전에 거부됩니다.
- 서로 다른 repetition은 서로 다른 승인 `contentText`를 가질 수 있습니다. repetition 간 Grounding 결과의 차이는 실험 변동의 일부로 봅니다.
- Quiz는 반드시 자기 repetition의 승인 Grounding을 원천으로 사용합니다. 다른 repetition의 Grounding을 `--source-grounding-run-id`로 지정하면 실행기가 Provider 호출 전에 거부하며, Quiz 결과 행도 고정 입력 manifest 항목도 만들지 않습니다.
- Direct C는 Grounding 없이 repetition마다 따로 실행합니다.
- offline 집계, Judge, Human blind evaluation은 Grounding과 Quiz의 repetition이 같을 것을 요구합니다.

### 완료와 재실행 규칙

Grounding:

- 조건(`videoId`·method·model·repetition·`video-grounding-v3`)당 성공한 Grounding 후보는 하나입니다. 성공한 Grounding이 하나라도 있으면 검수 상태(미검수·approved·rejected)와 관계없이 그 조건의 새 Grounding 생성은 Provider 호출 전에 거부됩니다.
- 다음 technical attempt는 `apiStatus=error`(구조적으로 사용할 수 없는 응답의 `invalid_grounding_response` 포함) 뒤에만 실행할 수 있습니다.
- rejected는 품질 판정이지 technical failure가 아닙니다. rejected Grounding으로는 Quiz A/B를 실행하지 않고, Grounding을 자동으로 다시 생성하지도 않습니다. 재생성이나 후보 선택은 별도의 사람 결정 사항입니다. 이 경우 그 영상·repetition의 Quiz A/B는 실행되지 않은 상태로 남습니다.

Quiz A/B·Direct C:

- 조건에 `apiStatus=success`가 한 번 기록되면 그 조건은 끝납니다. 이후 parsing 실패, 공통 구조 실패, `questionCount != 3`, evidence 검사 실패가 있어도 다시 생성하지 않습니다. 3문항 요구는 재생성 조건이 아니라 신뢰성 측정 대상입니다.
- `apiStatus=error`(technical failure)인 경우에만 다음 technical attempt를 실행할 수 있습니다. 한 technical attempt 안의 HTTP retry는 없습니다(`--retry-attempts 0`).

공통:

- 결과 행이 생성되지 않은 실행(승인·repetition·SHA-256·고정 입력·terminal 검사 거부, CLI 인자 오류)은 technical attempt가 아닙니다.
- 조건(같은 영상·repetition·조건)당 technical attempt는 최대 3회입니다(USER DECISION). technical attempt는 별도의 CLI 실행 한 번이며, 한 실행 안의 HTTP retry(`--retry-attempts`, `--call-limit`)와는 다른 단위입니다. 2·3번째 attempt는 앞선 attempt가 모두 `apiStatus=error`일 때만 실행하고, 첫 `apiStatus=success`가 나오면 3회 미만이어도 그 조건은 끝납니다. 따라서 parsing·구조·문항 수 실패를 3번까지 다시 생성한다는 뜻이 아닙니다. 3회 모두 `apiStatus=error`이면 "3 technical attempts incomplete"로 기록하고 중단하며, 이는 품질 실패가 아닙니다.
- 이 상한은 실행기가 자동으로 막지 않습니다. 실행 절차상 사람이 지켜야 하는 상한이므로, 다음 attempt 전에 그 조건의 기존 attempt 수를 결과 행에서 확인합니다.

### 영상·repetition별 실행 순서

영상 하나와 repetition 하나(`--repetition 1` 또는 `2`)마다 다음 순서로 실행합니다. 아래 명령은 공통 옵션(`--config configs/pilot-v2.yaml --results-dir results/pilot-v2-r2 --live`, 위 CLI 값, 가격 옵션)을 생략한 형태입니다.

1. Grounding: `python -m src.pilot_runner grounding --video-id <videoId> --repetition <r> --method gemini_video ...`
2. Grounding 검수: 사람이 영상과 `results/pilot-v2-r2/evaluation/<runId>.json`의 `contentText`·facts를 확인하고 다섯 항목(`factualAccuracy`, `keyInformationCoverage`, `factsConsistency`, `koreanConsistency`, `contentTextContractCompliance`)을 각각 `pass`·`fail`·`uncertain`으로 판정합니다. 판정의 의미는 [평가 기준](rubric.md)을 따릅니다. 판정은 전용 CLI가 없으므로 Python에서 기록합니다.

   ```python
   from src.pilot_runner import PilotRunner
   runner = PilotRunner(".", "results/pilot-v2-r2", "configs/pilot-v2.yaml")
   runner.review_grounding("<Grounding runId>", "taeyeonon",
                           {"factualAccuracy": "pass", "keyInformationCoverage": "pass",
                            "factsConsistency": "pass", "koreanConsistency": "pass",
                            "contentTextContractCompliance": "pass", "reviewNote": None})
   ```

   다섯 항목이 모두 `pass`면 approved, 하나라도 `fail`·`uncertain`이면 rejected입니다. 저장된 검수는 덮어쓰지 않습니다. `video-grounding-v3` Grounding은 `approve_content`로 승인할 수 없습니다. rejected면 그 영상·repetition의 3~6단계를 건너뛰고 7단계 Direct C로 갑니다.
3. approved면 그 Grounding의 evaluation `contentText`를 내용 변경 없이 Git-ignore된 로컬 content file(`data/restricted/` 아래, 영상·repetition마다 별도 파일)로 준비합니다. 끝 줄바꿈을 포함해 어떤 문자도 추가하거나 빼지 않습니다.
4. 첫 Quiz 실행 직전 고정 입력 재확인(아래 목록)
5. Quiz A: `python -m src.pilot_runner quiz --video-id <videoId> --repetition <r> --model gemini-3.8-flash --content-file <file> --source-grounding-run-id <Grounding runId> ...`
6. Quiz B: 5단계와 같은 `--repetition`, `--content-file`, `--source-grounding-run-id`로 `--model gpt-5.4-mini` 실행
7. Direct C: `python -m src.pilot_runner end-to-end --video-id <videoId> --repetition <r> --method gemini_direct_quiz ...`. Direct C는 Grounding 결과와 관계없이 실행합니다.

모든 영상·repetition의 생성이 끝난 뒤:

8. 자동 검사: parsing, 공통 구조, 문항 수, evidence 포함 여부는 실행 시 결과 행에 기록됩니다. Judge는 선택 사항인 보조 평가이며, 실행한다면 [Judge 평가 프로토콜](judge-protocol.md)에 따라 `--results-dir results/pilot-v2-r2`로 실행합니다.
9. Human blind evaluation(아래 "Human Evaluation" 참고)

첫 Quiz 실행 직전 고정 입력 재확인(각 영상·repetition의 첫 Quiz A 또는 B 전에 사람이 확인):

- `videoId`와 `--repetition`
- 명령의 `--source-grounding-run-id`가 이번 repetition에서 승인한 Grounding `runId`이고, 그 Grounding 행의 `repetition`이 명령의 `--repetition`과 같음
- 그 Grounding 행이 `promptVersion=video-grounding-v3`, `contentTextApprovalStatus=approved`이고 `groundingReview` 다섯 항목이 모두 `pass`이며 `approvedBy`·`approvedAt`이 기록됨
- evaluation `contentText`의 SHA-256이 Grounding 행의 `contentTextSha256`과 같음
- 명령의 `--content-file`이 이번 영상·repetition의 파일이고, 그 내용의 SHA-256이 같은 승인 해시와 같음
- `--config configs/pilot-v2.yaml`, `--results-dir results/pilot-v2-r2`, `--retry-attempts 0`

source `runId`나 content file이 잘못됐다면 Provider 호출 전에 바로잡고 다시 확인합니다. 첫 Quiz 실행 이후에는 A/B 결과를 보고 `contentText`를 바꾸지 않습니다.

### Human Evaluation

Pilot v1과 Pilot v2의 사람 평가 절차는 다릅니다.

- Pilot v1: 2장의 summary JSONL 수동 기록 절차와 `approve_content`입니다. Pilot v1 결과의 기록으로 유지합니다.
- Pilot v2 Grounding: 위 2단계의 `review_grounding` 체크리스트로만 승인·거부를 기록합니다. 결과는 Grounding 행의 `groundingReview`와 `contentTextApprovalStatus`에 남습니다.
- Pilot v2 Quiz A/B·Direct C: `src.human_quiz_evaluation`으로 blind evaluation을 합니다. 결과의 기준은 `results/pilot-v2-r2/human/<sessionId>/evaluation.json`이며, Pilot v2 Quiz 결과 행의 `questionReviews`에는 사람 판정을 기록하지 않습니다.

blind evaluation 절차:

1. 세션 생성: `python -m src.human_quiz_evaluation --results-dir results/pilot-v2-r2 create [--seed <seed>]`. 출력의 `sessionId`와 `blindExport` 경로를 기록합니다. seed를 생략하면 생성해 저장합니다.
2. `results/pilot-v2-r2/human/<sessionId>/blind-export.json`만 평가자에게 전달합니다. 여기에는 YouTube URL과 문항만 있고 조건·모델·runId·`contentText`·repetition은 없습니다. 같은 폴더의 `session.json`은 blindId와 원래 실행의 대응표이므로 평가자에게 주지 않습니다.
3. 평가자는 세트마다 `coverage`·`redundancy`·`learningValue`, 문항마다 여섯 항목을 `pass`·`fail`·`uncertain`으로 모두 채웁니다. `reviewNote`는 선택입니다. 판정 기준은 [평가 기준](rubric.md)을 따릅니다.
4. 가져오기: `python -m src.human_quiz_evaluation --results-dir results/pilot-v2-r2 import --session-id <sessionId> --evaluation-file <작성한 파일> --reviewed-by taeyeonon`
5. 세션마다 한 번만 가져올 수 있으며 저장된 `evaluation.json`은 덮어쓰지 않습니다. 다시 평가하려면 새 세션을 만듭니다.

세션은 설정된 영상 4개 × repetition 2 × A/B/C의 모든 조건을 다룹니다. 평가할 수 없는 조건은 `session.json`의 `notEligible`에 이유와 함께 남습니다.

- `notRun`: 기록된 실행이 없습니다. Grounding이 rejected되어 실행하지 않은 Quiz A/B나 아직 실행하지 않은 조건이 여기에 해당합니다. 모델의 생성 실패가 아니므로 Quiz 생성 신뢰성 실패로 해석하지 않습니다.
- `noApiSuccess`: 실행이 하나 이상 기록됐지만 `apiStatus=success`가 없습니다(technical failure만 있음).
- `parseFailed`, `questionCountMismatch`, `invalidQuestionStructure`: API 응답은 받았지만 평가자에게 보여줄 수 없는 출력입니다. 출력 신뢰성 결과로 집계합니다. evidence 검사 실패(`validatorStatus=fail`)만으로는 평가에서 제외하지 않습니다.

`reviewed_by`의 `taeyeonon`은 사람이 제공한 review tracking identifier이며 인증 정보가 아닙니다. 현재 검증 규칙(앞뒤 공백 제거 후 1~100자, 제어·서식 문자 없음)에 맞습니다. 실제 실행 전에 사용할 identifier를 다시 확인합니다.

### 실행 기록

각 실행의 실제 명령(API key 등 비밀값 제외), 실행 시각, 실제 실행 commit(checkout한 HEAD)을 남깁니다. 가격 확인 결과와 Human blind evaluation의 `sessionId`도 함께 기록합니다.

## 7. Pilot v2 결과 해석 기준

이 장은 Pilot v2 결과를 해석할 기준입니다. 실제 Pilot v2-r2 집계 결과와 이 기준에 따른 결정은 8장에 기록합니다.

- **표본 크기**: 영상 4개 × repetition 2입니다. 이 결과만으로 모델 전체의 성능을 일반화하지 않습니다.
- **Quiz A와 B**: 같은 repetition 안에서 같은 승인 `contentText`를 쓰므로 생성 모델 비교에 가장 가깝습니다. 다만 두 Provider의 `medium`은 같은 추론량을 뜻하지 않고, 구조화 출력 schema 적용 방식도 Provider마다 다릅니다. A/B 차이는 모델과 Provider별 설정을 합친 차이로 해석합니다.
- **Quiz A와 Direct C**: 차이를 Grounding 유무만의 인과 효과로 해석하지 않습니다. 입력(사람이 승인한 텍스트와 영상), Grounding 단계와 사람 검수, evidence 포함 검사 적용 여부가 모두 다른 pipeline 간 비교입니다.
- **repetition 간 차이**: Quiz A/B의 repetition 차이에는 Grounding 결과의 차이와 Quiz 생성의 차이가 함께 들어 있습니다. Direct C의 repetition 차이는 생성 자체의 차이입니다.
- **신뢰성 지표**: API 성공, parsing, 3문항 여부, 구조 검사를 사람 품질 판정과 구분해 기록합니다. `notRun`은 생성 실패로 세지 않고 실행하지 않은 조건으로 따로 보고합니다. Grounding rejection으로 Quiz A/B가 `notRun`이 된 경우는 Grounding 단계의 결과로 기록합니다.
- **sourceEvidence와 blind 평가**: blind export는 조건·모델·runId·`contentText`·repetition을 숨기지만, 문항의 `sourceEvidence`는 `evidenceSupportsAnswer` 판정에 필요해 그대로 보여줍니다. Quiz A/B의 `sourceEvidence`는 한국어 `contentText`를 인용하고 Direct C는 영상 내용을 직접 인용하므로, 문체로 A/B와 C가 구별될 수 있습니다. 평가자는 `contentText`를 보지 않으므로 근거 판정은 영상을 기준으로 합니다. 이 한계를 결과와 함께 기록합니다.
- **Judge**: 보조 평가이며 Human Evaluation이 기준입니다. Judge는 Pilot v2의 `evaluation.json`을 읽지 않으므로, Judge와 Pilot v2 Human 판정의 비교는 자동으로 계산되지 않습니다.
- **Pilot v1과의 비교**: prompt 버전, 생성 설정, 사람 평가 절차가 다르므로 Pilot v1 결과와 같은 조건의 결과로 합치지 않습니다.

## 8. Pilot v2-r2 결과와 Quiz 생성 방식 결정

### 실행 범위 (OBSERVED IN PILOT)

- 결과 디렉터리 `results/pilot-v2-r2`, 설정 `configs/pilot-v2.yaml`, 실행일 2026-10-07(UTC). 기준 commit은 Issue #31 수정이 반영된 `617d6ebd6c0c6864056fce5fa0ed56ca2764aa30`(PR #32)입니다. water·methane 실행 직전 HEAD가 이 commit임을 확인했습니다. mars·kari의 실제 실행 commit은 현재 저장된 저장소 근거만으로는 확인할 수 없습니다. 결과 행에는 실행 commit이 저장되지 않으며, 현재 HEAD나 결과 파일 시각으로 추정하지 않습니다.
- 영상 4개 × repetition 2 × 조건 A/B/C = Quiz set 24개, 문항 72개. Grounding 8건과 Quiz·Direct 24건, 모두 32건의 실행이 각각 attempt 1에서 `apiStatus=success`로 끝났습니다. 저장된 추정 비용 합계는 $0.3301입니다.
- Grounding 8건은 모두 `video-grounding-v3`이며, 사람이 원본 영상과 대조해 5개 검수 항목을 모두 `pass`로 기록했습니다(`approved`).
- Human blind evaluation은 세션 1개로 24개 set·72문항을 모두 평가했습니다(`notEligible` 없음).
- 두 번째 72문항 실행은 하지 않습니다(USER DECISION). 이 프로토콜이 정한 표본을 모두 채웠습니다.

### 비교한 조건

| 조건 | 흐름 | 모델·방식 | 생성 설정 |
| --- | --- | --- | --- |
| A | Gemini Grounding → 사람 검수·승인 → 고정 `contentText` → Quiz | `gemini-3.8-flash`, `fixed_content_text`, `pilot-v2` | `thinking_level: medium` |
| B | A와 같은 repetition의 같은 승인 `contentText` → Quiz | `gpt-5.4-mini`, `fixed_content_text`, `pilot-v2` | `reasoning_effort: medium` |
| C | YouTube 영상 → 영상 이해와 Quiz 생성을 한 번에 | `gemini-3.8-flash`, `gemini_direct_quiz`, `pilot-v2` | `thinking_level: medium` |

A/B의 Grounding은 `gemini-3.8-flash`, `gemini_video`, `video-grounding-v3`(thinking 설정 없음)이며 repetition마다 하나를 A와 B가 공유합니다. C에는 별도의 Grounding 단계가 없습니다.

### 결과 (OBSERVED IN PILOT)

Human blind evaluation(조건별 8 set·24문항, set 단위 3항목과 문항 단위 6항목):

| 조건 | set | 문항 | 판정 |
| --- | --- | --- | --- |
| A | 8 | 24 | 168개 모두 `pass` |
| B | 8 | 24 | 168개 모두 `pass` |
| C | 8 | 24 | 168개 모두 `pass` |

이번 Pilot 표본에서 Human 평가 기준으로 조건 간 차이는 관측되지 않았습니다. 세 방식의 일반적인 품질이 같다는 증명은 아닙니다(7장 "표본 크기").

자동 신뢰성 지표:

| 지표 | A | B | C |
| --- | --- | --- | --- |
| API 성공 | 8/8 | 8/8 | 8/8 |
| parsing | 8/8 | 8/8 | 8/8 |
| 3문항 | 8/8 | 8/8 | 8/8 |
| validator | 8/8 `pass` | 7/8 `pass`, 1/8 `fail` | 8/8 `not_run` (해당 없음) |
| `beCompatibility` | 8/8 `pass` | 7/8 `pass`, 1/8 `fail` | 8/8 `not_applicable` |

- B의 1건(methane repetition 1, runId `49966005f167411394bc5d91a399059d`, `evidence_not_in_content`): 세 `sourceEvidence`가 모두 ASCII 큰따옴표로 감싸여 있어 정확한 부분 문자열 계약을 통과하지 못했습니다. 따옴표 안의 문장은 승인 `contentText`와 정확히 일치하며, 이 set은 Human 평가에서 `pass`였습니다. 의미상 Human 품질 실패가 아니라 현재 evidence 직렬화·정확 부분 문자열 계약과의 불일치 1건입니다. v2 규칙에 따라 다시 생성하지 않았습니다.
- C에는 고정 `contentText`가 없어 A/B의 evidence validator를 적용하지 않습니다. C의 `validatorStatus=not_run`과 `beCompatibility=not_applicable`은 통과도 실패도 아닙니다. C는 parsing과 공통 구조 검사를 모두 통과했습니다.

### 비용과 지연 해석

비용은 실행기가 실제 사용량과 실행 직전 확인한 공식 단가로 계산한 추정치(`estimatedCostUsd`)이며 실제 청구액이 아닙니다. 지연은 Provider 응답 시간이며, 사람의 Grounding 검수 시간은 포함하지 않습니다.

| 지표 (조건별 평균) | A | B | C |
| --- | --- | --- | --- |
| Quiz 단계만 비용 | $0.00456 | $0.00414 | $0.01670 (영상 처리 포함) |
| Grounding 포함 단순 순차 비용 | $0.02042 | $0.02001 | $0.01670 |
| Quiz 단계만 지연 | 9.95초 | 6.26초 | 20.79초 (영상 처리 포함) |
| Grounding 포함 단순 순차 지연 | 26.82초 | 23.12초 | 20.79초 |

- Grounding 평균 비용은 $0.01587, 평균 지연은 16.87초입니다. A/B의 Quiz 단계 수치에는 이 Grounding이 포함되지 않습니다.
- 따라서 Quiz 단계 수치만으로 "C가 비싸다·느리다"고 비교하지 않습니다. "영상 하나 → Quiz set 하나" 흐름에서 Grounding까지 더하면 C가 비용·지연 면에서 불리하다는 결과는 나오지 않았습니다.
- Grounding 하나를 여러 Quiz 생성에 재사용하는 구조라면 A/B의 분할 비용·지연은 달라집니다. 조건별 8건의 평균이며, 영상과 시점에 따라 달라질 수 있습니다.

### Quiz 생성 방식 결정: C — Gemini Direct (USER DECISION, 이후 변경)

> 이 결정은 이후 [Production architecture 재검토](#production-architecture-재검토-a-기반-구조로-변경-user-decision)에서 A 기반 구조로 변경되었습니다. 결정 이력으로 이 소절과 다음 두 소절의 원래 기록을 유지합니다. 다음 소절의 production 목표 흐름과 Judge 방향의 생성기 표기(`gemini_direct_quiz`)는 변경 전 기록입니다.

Production Quiz 생성의 기본 방향으로 C(YouTube 영상 → `gemini-3.8-flash` `gemini_direct_quiz` → Quiz)를 선택합니다. 근거:

1. **Human 품질**: 이번 Pilot에서 A/B/C 모두 Human 평가 전 항목이 `pass`였고, C에서 A/B보다 낮은 품질은 관측되지 않았습니다.
2. **서비스 흐름 일치**: 서비스 흐름은 Creator가 YouTube 링크 입력 → Quiz 생성 → 생성된 Quiz 확인 → Creator 승인 → 게시입니다. C는 이 한 번의 생성 흐름과 가장 직접적으로 일치합니다.
3. **pipeline 단순성**: A/B는 영상 → Grounding → 고정 `contentText` → Quiz의 중간 단계가 필요하고, C는 영상 → Quiz입니다.
4. **E2E 비용**: C의 비용에는 영상 처리가 포함되어 A/B의 Quiz 단계 비용과 직접 비교할 수 없으며, Grounding까지 포함한 단순 비교에서 C가 불리하지 않았습니다.
5. **E2E 지연**: Grounding 지연까지 고려하면 한 번의 생성 흐름에서 C는 경쟁력이 있습니다.
6. **최종 승인 위치**: Creator가 생성된 Quiz를 확인하고 승인하므로, 중간 Grounding에 대한 사람 승인을 추가하기보다 최종 Quiz의 품질 검증과 Creator 승인에 집중합니다.

이 결정은 다음을 주장하지 않습니다: C의 Quiz 품질이 A/B보다 좋다는 것(이번 Human 결과는 동률), C가 항상 더 싸거나 빠르다는 것, C가 모든 영상에서 더 우수하다는 것. Human 품질이 이번 Pilot에서 동등하게 관측된 상황에서 서비스 흐름, pipeline 단순성, E2E 비용·지연 구조를 함께 고려한 선택입니다.

### Pilot 프로토콜과 production의 구분

Pilot의 Grounding 생성 → 사람 Grounding 검수 → 승인 → 고정 `contentText` → Quiz 절차는 A/B/C를 공정하고 검증 가능하게 비교하기 위한 Pilot 프로토콜의 일부입니다. production의 필수 사람 절차가 아니며, production에서 Creator가 중간 Grounding을 검토·승인하는 UX는 현재 목표가 아닙니다.

Production 목표 흐름:

```
YouTube 링크 → Gemini Direct Quiz 생성 → GPT 계열 LLM Judge → Creator 확인·승인 → 게시
```

### Judge 방향 (USER DECISION)

- Quiz 생성기는 Gemini 계열(현재 선택: `gemini-3.8-flash`, `gemini_direct_quiz`), Quiz 검증용 LLM Judge는 GPT 계열을 사용합니다.
- 이는 architecture 결정입니다. 생성기와 Judge를 서로 다른 model·provider 계열로 분리해, 생성 모델이 자기 출력을 다시 평가하는 구조를 피하기 위한 것입니다. GPT가 Gemini보다 Judge 성능이 좋다는 Pilot 결과나 Judge 모델 비교 실험에 근거한 결론이 아닙니다.
- 결정됨: Judge 계열 GPT.
- 후속 결정: 정확한 GPT Judge 모델, Judge prompt version, 판정 기준(threshold)과 pass/fail 정책, retry 정책, reject 후 재생성 정책, Judge 실패 시 fallback, Direct Quiz 입력 계약, production 비용 상한.
- 현재 저장소의 Judge([Judge 평가 프로토콜](judge-protocol.md))는 A/B를 승인 `contentText` 기준으로 사후 평가하는 보조 도구이며, Direct Quiz(C)는 범위 밖입니다. 따라서 C에 대한 production GPT Judge는 입력 계약을 포함해 별도로 설계해야 하며, 기존 Judge를 그대로 쓸 수 있는지는 후속 조사 대상입니다.

### Production architecture 재검토: A 기반 구조로 변경 (USER DECISION)

Production Quiz 생성기 구조를 C(Gemini Direct)에서 A 기반 구조로 변경합니다. Pilot 결과를 새로 해석하거나 추가 실험을 한 결과가 아니라, production Judge를 붙이는 방법을 검토하면서 내린 architecture 결정입니다.

결정 순서:

1. **Pilot 결과**: A/B/C 모두 Human 평가 전 항목 `pass`로 조건 간 품질 차이가 관측되지 않았습니다(위 "결과").
2. **최초 선택 C**: 서비스 흐름 일치, pipeline 단순성, E2E 비용·지연 구조를 근거로 C를 선택했습니다(위 "Quiz 생성 방식 결정: C").
3. **PR #20 Judge 재분석**: production GPT Judge를 C에 붙일 수 있는지 현재 Judge 코드와 계약을 다시 확인했습니다.
4. **Direct C의 evidence blocker**: C에는 Judge가 기준으로 삼을 독립된 근거가 없습니다.
   - C의 `sourceEvidence`는 Quiz를 생성한 모델 자신의 출력입니다. 생성과 독립적으로 만들어진 evidence 산출물이나 ground-truth 계약이 없습니다.
   - 현재 Judge 계약은 승인된 `contentText`만을 판정 근거로 사용하며(`src/judge_contract.py`), C는 범위 밖입니다([Judge 평가 프로토콜](judge-protocol.md)).
   - 현재 OpenAI Judge adapter는 prompt 텍스트만 입력으로 보내며(`src/judge_client.py`의 `build_request`) 영상 입력 경로가 없습니다. 따라서 GPT Judge가 C의 Quiz를 영상 기준으로 판정할 수 없습니다.
5. **A 기반 구조로 변경**: Quiz 생성 전에 Gemini Grounding이 `contentText`를 만들고, Quiz와 Judge가 같은 `contentText`를 기준으로 삼는 구조로 바꿉니다.
6. **GPT Judge 계열 유지**: Quiz 생성기(Gemini)와 Judge(GPT)를 서로 다른 계열로 분리하는 Judge 방향(위 "Judge 방향")은 그대로입니다.
7. **evidence 신뢰 정책은 다음 설계 과제**: 자동 Grounding 결과를 어디까지 믿고 어떻게 다룰지는 아직 정하지 않았습니다.

변경된 production 목표 흐름:

```
YouTube 링크 → Gemini 자동 Grounding → contentText → Gemini Quiz 생성 → GPT Pointwise Judge → Creator 확인·승인 → 게시
```

A 기반 구조를 택한 이유:

- **명시적인 evidence 산출물**: `contentText`가 Quiz 생성과 분리된 별도 산출물로 남습니다. Quiz의 `sourceEvidence`를 이 텍스트와 대조할 수 있습니다.
- **기존 Pointwise Judge 구성요소 재사용**: OpenAI transport, strict structured output parser, schema 검증, retry와 재개(resume), usage·비용 계산, 요청 journal과 결과 저장, `contentText` 기준 Pointwise 구조를 이어서 쓸 수 있습니다.
- **감사 가능성(auditability)**: Quiz와 Judge 판정이 어떤 텍스트를 근거로 했는지 저장된 `contentText`로 추적할 수 있습니다.
- **production Judge 호환성**: 텍스트 기준 Judge 계약과 text-only OpenAI Judge 입력 경로를 그대로 따를 수 있습니다.
- **Pilot 품질 결과와 충돌하지 않음**: 이번 Pilot에서 A의 Human 판정은 C와 같이 모두 `pass`였으므로, A 기반 구조를 선택해도 관측된 품질 결과와 어긋나지 않습니다.

재사용 범위의 한계: 현재 Judge는 Pilot의 A/B 조건을 대상으로 하며, 원천 Grounding이 사람 승인(`contentTextApprovalStatus=approved`와 승인 추적 필드)을 갖춘 경우에만 대상으로 삼습니다(`src/judge_eligibility.py`). production의 자동 Grounding 결과는 이 조건을 만족하지 않으므로, 현재 Judge 실행기를 그대로 production에 쓸 수 있는 것은 아닙니다. 재사용 대상은 위의 구성요소이며, 자동 Grounding evidence 입력 계약은 새로 정해야 합니다.

분명히 해 둘 점:

- Pilot의 사람 Grounding 검수·승인은 production workflow에 포함되지 않습니다. Pilot 비교를 위한 프로토콜이었습니다(위 "Pilot 프로토콜과 production의 구분").
- production의 자동 Grounding 결과는 검증된 ground truth가 아닙니다. Judge가 `contentText` 기준으로 `pass`를 내도 영상 기준의 사실 정확성을 보장하지 않습니다.
- 정확한 GPT Judge 모델은 정하지 않았습니다(TBD). `configs/judge.yaml`의 `gpt-6.1-sol`은 Pilot 사후 Judge 설정이며 production 모델 결정이 아닙니다.
- production에는 Pairwise가 필요하지 않습니다. production Judge는 Pointwise입니다.
- Creator의 최종 확인·승인은 유지합니다.
- 두 번째 72문항 실행은 하지 않습니다(위 "실행 범위").

트레이드오프:

- 영상 → Quiz 사이에 Grounding 단계가 하나 더 생깁니다.
- Grounding 호출이 하나 늘어 비용과 지연이 커질 수 있습니다. 이번 Pilot의 단순 순차 합산 평균에서 A는 C보다 비용·지연이 높았습니다(위 "비용과 지연 해석", production 수치가 아님).
- Grounding 실패가 새로운 실패 지점이 됩니다. Grounding이 실패하면 Quiz를 생성할 수 없으며, production의 Grounding retry·실패 처리 정책은 정하지 않았습니다.
- Grounding의 오류가 Quiz와 Judge evidence 양쪽으로 전파됩니다. Judge는 같은 `contentText`를 기준으로 판정하므로, `contentText` 자체의 사실 오류는 Judge가 잡아내지 못할 수 있습니다.
- 사람 검수 없이 자동 Grounding 결과를 쓰므로 evidence 신뢰 정책이 필요합니다.

이 변경은 다음을 주장하지 않습니다: A의 품질이 C보다 좋다는 것, C가 실패했다는 것, A/C 비교에서 A가 이겼다는 것, GPT Judge 실험 결과가 이 결정을 뒷받침한다는 것. 이 결정은 Judge 실행 결과에 근거하지 않습니다.

다음 단계 (제안, 아직 진행하지 않음):

1. Production A 자동 Grounding evidence 계약 정의(evidence 신뢰 정책 포함).
2. Production A와 GPT Pointwise Judge 지원 구현.
3. GPT Judge 검증: Human 평가로 `pass`가 확인된 기존 Quiz를 positive 데이터로 재사용하고, 알려진 결함 사례(known-bad)를 추가해 확인.
4. 최종 E2E 검증.

## 공식 문서 출처 (2026-10-02 확인)

- Gemini: [Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding), [Video understanding (Interactions API)](https://ai.google.dev/gemini-api/docs/interactions/video-understanding), [Models](https://ai.google.dev/gemini-api/docs/models), [Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- Azure Content Understanding: [Video overview](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/video/overview), [Service quotas and limits](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/service-limits), [Region and language support](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/language-region-support), [Data, privacy, and security](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/content-understanding/data-privacy), [Pricing](https://azure.microsoft.com/en-us/pricing/details/content-understanding/)
- TwelveLabs: [Analyze videos](https://docs.twelvelabs.io/docs/guides/analyze-videos), [Pegasus](https://docs.twelvelabs.io/docs/concepts/models/pegasus), [Pricing](https://www.twelvelabs.io/pricing), [Terms of Use](https://www.twelvelabs.io/legal/terms-of-use)

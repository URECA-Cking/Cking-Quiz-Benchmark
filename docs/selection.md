# 방식·모델 선택 기록

이 문서는 Cking AI Quiz Pilot에서 영상 분석 방법을 정한 근거와 현재 Pilot의 실행 프로토콜을 기록합니다. 근거의 성격은 다음 표기로 구분합니다.

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

## 2. 현재 Pilot 실행 프로토콜 (CURRENT PILOT POLICY)

### 범위

- 현재 Pilot 단계에서는 4개 영상의 repetition 1을 완료합니다.
- 첫 영상 `nasa-water-cycle-2019`는 기존 결과를 유지합니다(3장 참고).
- 남은 3개 영상 `nasa-methane-2020`, `nasa-mars-organics-2025`, `kari-microgravity-2024`에 이 프로토콜을 적용합니다.
- repetition 2는 현재 범위에서 실행하지 않습니다. 고정 입력 manifest(영상·`promptVersion`당 `contentText` 하나)와 aggregator(Grounding·Quiz의 repetition 일치 요구) 계약 때문에, repetition 1 완료 후 의미와 계약을 별도로 결정합니다.

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

결과를 선택 근거로 사용할 때 Video Grounding, 고정 입력 Quiz Generation, End-to-End 결과를 구분해 기록합니다. 각 선택에는 사용 데이터와 권리 확인, 실행 날짜, 모델 ID, 프롬프트 버전, 가격 출처, 자동 지표, 실제 영상과 승인된 ground truth를 대조한 사람 검수 근거, 제한사항 및 Cking-BE 적용 결정을 남깁니다. 직접 Quiz의 `beCompatibility`는 `not_applicable`입니다. 다른 영상의 Pilot과 최종 Quiz 방식·모델 선정은 아직 완료되지 않았습니다.

## 5. 결과 해석 caveat와 후속 과제

다음은 현재 Pilot의 blocker가 아닙니다.

결과 해석 시 함께 기록할 caveat:

- Quiz A와 Direct C의 차이는 분리 방식과 직접 방식의 차이만이 아닙니다. 입력(승인된 텍스트와 영상), 사람 검수·승인 단계, evidence 포함 검사 적용 여부도 다릅니다.
- Quiz A와 B는 같은 승인 `contentText`를 쓰지만, Gemini thinking 기본값과 OpenAI `reasoning-effort low`, Provider별 구조화 출력 schema 적용 방식이 다릅니다.

별도 코드 후속 과제(이번 Pilot 프로토콜에 포함하지 않음):

- offline aggregator가 선택된 Quiz의 `parseStatus`·`validatorStatus`·`errorCategory` 등을 충분히 노출하지 않음
- timestamp 시작·끝과 영상 길이 범위 검증의 제한
- 중복 `runId`의 전역 탐지 제한
- NaN·Infinity 처리 계약의 경로별 차이
- 전역 storage preflight의 승인 추적 검증 범위 제한
- 실행 설정 전체 snapshot 부재

## 공식 문서 출처 (2026-10-02 확인)

- Gemini: [Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding), [Video understanding (Interactions API)](https://ai.google.dev/gemini-api/docs/interactions/video-understanding), [Models](https://ai.google.dev/gemini-api/docs/models), [Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- Azure Content Understanding: [Video overview](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/video/overview), [Service quotas and limits](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/service-limits), [Region and language support](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/language-region-support), [Data, privacy, and security](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/content-understanding/data-privacy), [Pricing](https://azure.microsoft.com/en-us/pricing/details/content-understanding/)
- TwelveLabs: [Analyze videos](https://docs.twelvelabs.io/docs/guides/analyze-videos), [Pegasus](https://docs.twelvelabs.io/docs/concepts/models/pegasus), [Pricing](https://www.twelvelabs.io/pricing), [Terms of Use](https://www.twelvelabs.io/legal/terms-of-use)

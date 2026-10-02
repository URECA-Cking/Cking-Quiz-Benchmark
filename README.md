# Cking Quiz Benchmark

AI 퀴즈의 영상 근거 확보 방식과 생성 모델을 **별도로** 비교하기 위한 실험 저장소입니다. 운영 API와 데이터 저장은 [Cking-BE](https://github.com/URECA-Cking/Cking-BE)의 책임입니다. 이 저장소의 실험 결과만으로 운영 기능의 성공을 주장하지 않습니다.

## Pilot 범위

- 이용 조건을 확인한 공개 YouTube 영상 3~5개. 현재 후보는 영어 NASA 3개, 한국어 KARI 1개입니다.
- 영상당 객관식 3문제, 문제당 보기 4개, 조건별 반복 실행. 모델 ID·제공 상태·요금은 실행 직전에 다시 확인합니다.
- **Video Grounding Benchmark:** Gemini가 공개 YouTube 영상을 분석한 결과와 사용 권한을 확인한 transcript/caption의 근거를 각각 평가합니다. 사실의 영상 내 존재, 발화/화면 출처, timestamp 정확성, 누락·환각, 지연·토큰·비용을 기록합니다.
- **Quiz Generation Benchmark:** 영상별로 동일하게 고정한 `contentText`를 `gemini-3.8-flash`와 `gpt-5.4-mini`에 제공합니다. 같은 프롬프트 버전과 Quiz 계약으로 파싱·검증, 정답 정확성·유일성, 근거의 정답 지지 여부, 한국어 품질, 지연·토큰·비용을 비교합니다. Grounding을 반복 호출하지 않아 생성 모델만 비교할 수 있어야 합니다.
- **End-to-End Benchmark:** Gemini Grounding → Gemini/OpenAI Quiz, YouTube → Gemini 직접 Quiz, 권한 있는 transcript → Gemini/OpenAI Quiz를 비교합니다. transcript를 확보할 수 없는 영상은 해당 조건을 `not_run`으로 기록합니다.

기존 A/B/C와의 관계: **A**는 Gemini Grounding 후 Quiz 생성의 두 조합, **B**는 Gemini 직접 Quiz, **C**는 권한 있는 transcript 기반 두 조합입니다. 새로운 Quiz Generation Benchmark는 A/C의 한 실행을 그대로 비교하지 않고 **동일한 고정 입력**으로 생성 모델의 차이만 측정합니다.

현재는 평가 계약과 영상 후보, 사람 검수를 마친 ground truth, **fixture 및 명시적 live 단일 조건 실행기**가 있습니다. fixture 결과는 API 미실행(`apiStatus=not_run`)으로 기록하며 실제 latency·token·cost 성능 결과로 사용하지 않습니다. 첫 Pilot 영상 `nasa-water-cycle-2019`에서는 Grounding attempt 1~4 실패 후 attempt 5가 성공했고, 해당 `contentText`의 사람 승인과 동일 승인 입력을 공유한 Gemini Quiz A·OpenAI Quiz B의 실행 및 사람 평가를 마쳤습니다. A/B의 offline 집계가 가능하지만 Pilot 전체가 완료된 것은 아닙니다. Gemini Direct Quiz는 이 영상의 현재 조건에서 technical attempt 1~3이 완료되지 않았으며, 세 번째는 HTTP 503 / `service_unavailable`로 기록됐습니다. 원본 영상 파일과 transcript 전문은 저장하지 않습니다. Pilot을 검증한 뒤에만 영상 12~18개와 추가 모델로 확대합니다.

## 평가 원칙

Grounding 결과와 Quiz 결과는 별도로 저장하고 실행 ID로 연결합니다. 모델이 제시한 사실·시점·근거 문구는 실제 영상 및 사람 검수 ground truth와 대조합니다. timestamp가 있다는 이유만으로 정확하다고 판정하지 않습니다. API 성공, JSON 파싱, Benchmark 내부 Quiz 형식/계약 검사와 영상 기반 사실성 판단을 구분합니다. 세부 기준은 [평가 기준](docs/rubric.md)을 따릅니다.

`Cking-BE`의 입력 계약을 참고해 고정 `contentText` Quiz의 `sourceEvidence`가 입력 텍스트에 포함되는지 Benchmark 내부에서 검사합니다. 기존 필드명 `beCompatibility=pass`는 이 로컬 형식/계약 검사 통과만 뜻하며, 실제 Cking-BE validator·production DTO/domain validation 또는 서비스 integration test 통과를 뜻하지 않습니다. Gemini 직접 Quiz는 비교할 고정 `contentText`가 없어 `beCompatibility=not_applicable`입니다. 이 값도 품질이나 정답 정확성의 통과를 뜻하지 않습니다. 문자열 포함 검사는 해당 문장이 **실제 영상에 존재하거나 정답을 의미적으로 뒷받침함**을 증명하지 않습니다.

현재 [ground truth 후보](data/ground_truth_candidates.jsonl) 12개는 영상별 3개이며 모두 `approved`입니다. 사용자가 YouTube 영상의 자막·내용을 확인해 후보 사실과 일치한다고 판정했습니다. 이는 timestamp의 프레임 단위 정확성이나 원본 transcript 전문을 검증했다는 뜻은 아닙니다. 새 후보는 `pending_human`으로 등록하고 사람이 검수한 뒤에만 승인합니다.

## 데이터와 설정

- [`configs/pilot.yaml`](configs/pilot.yaml): 세 평가의 방식·모델·문제 수·반복 횟수 설정.
- [`data/videos.schema.json`](data/videos.schema.json): 영상 메타데이터와 사용 조건 기록 계약.
- [`data/ground_truth.schema.json`](data/ground_truth.schema.json): 사람이 확인한 사실·근거·구간의 기록 계약.
- [`data/videos.jsonl`](data/videos.jsonl): 공식 링크와 자막을 확인한 Pilot 영상 후보 4개(영어 3, 한국어 1).
- [`data/ground_truth_candidates.jsonl`](data/ground_truth_candidates.jsonl): 공식 자료에서 추린 사실 후보 12개. YouTube 자막·내용에 대한 사람 검수 후 모두 `approved`.
- [`docs/pilot-data-review.md`](docs/pilot-data-review.md): 권리·언어·영상 확인의 남은 조건.
- [`docs/results-format.md`](docs/results-format.md), [`docs/run-result.schema.json`](docs/run-result.schema.json): 계층별 실행 JSONL과 집계 CSV 계약.
- [`docs/selection.md`](docs/selection.md): 실험 뒤 선택 근거를 기록할 자리.
- [`src/README.md`](src/README.md): fixture 실행 방법과 후속 Provider 경계.

실제 API 키는 각각 `GEMINI_API_KEY`, `OPENAI_API_KEY` 환경변수로만 전달합니다. 키를 코드, 설정 파일, `.env`, 로그, Git에 저장하지 않습니다. 공개 저장소에는 출처를 확인한 메타데이터와 재서술한 사실 후보만 올립니다. 영상·자막 전문, 제한된 transcript, 원시 API 응답 및 로컬 결과는 Git에 포함하지 않습니다.

## 다음 단계

승인된 후보를 바탕으로 transcript 사용 권한, 실제 모델 ID와 비용 상한을 확정해야 합니다. 현재 실행기는 기본적으로 fixture만 읽으며, live 실행은 명시적 옵션·상한이 모두 지정될 때만 허용합니다. 실제 Pilot 실행 전 별도 검토가 필요합니다.

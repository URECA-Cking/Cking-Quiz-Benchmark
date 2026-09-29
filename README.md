# Cking Quiz Benchmark

AI 퀴즈의 영상 근거 확보 방식과 생성 모델을 **별도로** 비교하기 위한 실험 저장소입니다. 운영 API와 데이터 저장은 [Cking-BE](https://github.com/URECA-Cking/Cking-BE)의 책임입니다. 이 저장소의 실험 결과만으로 운영 기능의 성공을 주장하지 않습니다.

## Pilot 범위

- 이용 조건을 확인한 공개 YouTube 영상 3~5개. 현재 후보는 영어 NASA 3개, 한국어 KARI 1개입니다.
- 영상당 객관식 3문제, 문제당 보기 4개, 조건별 반복 실행. 모델 ID·제공 상태·요금은 실행 직전에 다시 확인합니다.
- **Video Grounding Benchmark:** Gemini가 공개 YouTube 영상을 분석한 결과와 사용 권한을 확인한 transcript/caption의 근거를 각각 평가합니다. 사실의 영상 내 존재, 발화/화면 출처, timestamp 정확성, 누락·환각, 지연·토큰·비용을 기록합니다.
- **Quiz Generation Benchmark:** 영상별로 동일하게 고정한 `contentText`를 `gemini-3.8-flash`와 `deepseek-flash`에 제공합니다. 같은 프롬프트 버전과 Quiz 계약으로 파싱·검증, 정답 정확성·유일성, 근거의 정답 지지 여부, 한국어 품질, 지연·토큰·비용을 비교합니다. Grounding을 반복 호출하지 않아 생성 모델만 비교할 수 있어야 합니다.
- **End-to-End Benchmark:** Gemini Grounding → Gemini/DeepSeek Quiz, YouTube → Gemini 직접 Quiz, 권한 있는 transcript → Gemini/DeepSeek Quiz를 비교합니다. transcript를 확보할 수 없는 영상은 해당 조건을 `not_run`으로 기록합니다.

기존 A/B/C와의 관계: **A**는 Gemini Grounding 후 Quiz 생성의 두 조합, **B**는 Gemini 직접 Quiz, **C**는 권한 있는 transcript 기반 두 조합입니다. 새로운 Quiz Generation Benchmark는 A/C의 한 실행을 그대로 비교하지 않고 **동일한 고정 입력**으로 생성 모델의 차이만 측정합니다.

현재는 평가 계약과 영상 후보만 있습니다. Gemini·DeepSeek 호출 코드, 원본 영상 파일, transcript 전문 및 성능 결과는 없습니다. Pilot을 검증한 뒤에만 영상 12~18개와 추가 모델로 확대합니다.

## 평가 원칙

Grounding 결과와 Quiz 결과는 별도로 저장하고 실행 ID로 연결합니다. 모델이 제시한 사실·시점·근거 문구는 실제 영상 및 사람 검수 ground truth와 대조합니다. timestamp가 있다는 이유만으로 정확하다고 판정하지 않습니다. API 성공, JSON 파싱, `Cking-BE` 계약 검사와 영상 기반 사실성 판단을 구분합니다. 세부 기준은 [평가 기준](docs/rubric.md)을 따릅니다.

`Cking-BE`의 현재 입력 계약은 `QuizGenerationInput(contentText)`이며 `sourceEvidence`가 해당 `contentText`에 포함되어야 합니다. `contentText`가 있는 A/C와 고정 입력 Quiz Benchmark에는 BE 호환 여부를 기록합니다. B는 원문 `contentText`가 없으므로 `beCompatibility`를 항상 `not_applicable`로 기록합니다. 문자열 포함 검사는 해당 문장이 **실제 영상에 존재하거나 정답을 뒷받침함**을 증명하지 않습니다.

현재 [ground truth 후보](data/ground_truth_candidates.jsonl) 12개는 영상별 3개이며 모두 `pending_human`입니다. 실제 영상을 직접 확인하고 검수자·시각을 기록하기 전에는 확정 ground truth로 사용하지 않습니다. 영상 4개와 후보 12개의 내용·상태는 이번 설계 변경에서 수정하지 않습니다.

## 데이터와 설정

- [`configs/pilot.yaml`](configs/pilot.yaml): 세 평가의 방식·모델·문제 수·반복 횟수 설정. 실행기는 아직 없습니다.
- [`data/videos.schema.json`](data/videos.schema.json): 영상 메타데이터와 사용 조건 기록 계약.
- [`data/ground_truth.schema.json`](data/ground_truth.schema.json): 사람이 확인한 사실·근거·구간의 기록 계약.
- [`data/videos.jsonl`](data/videos.jsonl): 공식 링크와 자막을 확인한 Pilot 영상 후보 4개(영어 3, 한국어 1).
- [`data/ground_truth_candidates.jsonl`](data/ground_truth_candidates.jsonl): 공식 자막에서 추린 사실 후보. 모두 사람 검수 전이며 최종 ground truth가 아님.
- [`docs/pilot-data-review.md`](docs/pilot-data-review.md): 권리·언어·영상 확인의 남은 조건.
- [`docs/results-format.md`](docs/results-format.md), [`docs/run-result.schema.json`](docs/run-result.schema.json): 계층별 실행 JSONL과 집계 CSV 계약.
- [`docs/selection.md`](docs/selection.md): 실험 뒤 선택 근거를 기록할 자리.
- [`src/README.md`](src/README.md): 후속 실행 코드의 책임 경계.

실제 API 키는 각각 `GEMINI_API_KEY`, `DEEPSEEK_API_KEY` 환경변수로만 전달합니다. 키를 코드, 설정 파일, `.env`, 로그, Git에 저장하지 않습니다. 공개 저장소에는 출처를 확인한 메타데이터와 재서술한 사실 후보만 올립니다. 영상·자막 전문, 제한된 transcript, 원시 API 응답 및 로컬 결과는 Git에 포함하지 않습니다.

## 다음 단계

후보 영상의 실제 재생 내용과 사실을 사람이 확인하고, transcript 사용 권한 및 비용 상한을 확정한 뒤 실행기를 구현합니다. 현재는 아직 Benchmark를 실행할 수 없습니다.

# Cking Quiz Benchmark

AI 퀴즈의 생성 방식과 모델을 비교하기 위한 실험 저장소입니다. 운영 API와 데이터 저장은 [Cking-BE](https://github.com/URECA-Cking/Cking-BE)의 책임입니다. 이 저장소의 실험 결과만으로 운영 기능의 성공을 주장하지 않습니다.

## Pilot 범위

- 이용 조건을 확인한 공개 YouTube 영상 3~5개
- 초기 모델 설정: `gemini-3.8-flash` (실행 전 공식 제공 상태와 정확한 모델 ID 재확인)
- 같은 영상·문제 수·보기 수·프롬프트 버전에서 A/B/C 방식 비교, 조건별 반복 실행
  - A: YouTube URL → 모델이 `contentText` 추출 → 퀴즈 생성
  - B: YouTube URL → 모델이 직접 퀴즈 생성
  - C: 사용 가능한 transcript/caption → `contentText` → 퀴즈 생성
- 영상당 객관식 3문제, 문제당 보기 4개

이 단계는 실행 골격과 평가 계약만 정의합니다. 아직 Gemini 호출 코드, 영상 데이터, transcript 및 성능 결과가 없습니다. Pilot 검증 이후에만 영상 12~18개와 추가 모델로 확대합니다.

## 평가 원칙

API 성공과 퀴즈 품질을 분리합니다. JSON 파싱, 문제·보기 수, 0-based 정답 인덱스, 근거 문자열 포함 여부는 자동 평가 후보입니다. 영상과의 관련성, 정답의 사실성, 실제 영상 근거, 환각과 모호성은 사람이 원본 영상 또는 권한 있는 원문을 확인해 평가합니다. 자세한 기준은 [평가 기준](docs/rubric.md)을 따릅니다.

`Cking-BE`의 현재 입력 계약은 `QuizGenerationInput(contentText)`이며 `sourceEvidence`가 해당 `contentText`에 포함되어야 합니다. A/C에는 이 계약과의 호환 여부를 따로 기록합니다. B는 원문 `contentText`가 없으므로 BE 검증 통과로 기록하지 않고 `not_applicable`로 표시합니다. 모델이 만든 설명이나 timestamp만으로 실제 영상 근거가 검증된 것으로 간주하지 않습니다.

## 데이터와 설정

- [`configs/pilot.yaml`](configs/pilot.yaml): Pilot의 방식·모델·문제 수·반복 횟수 설정. 실행기는 아직 없습니다.
- [`data/videos.schema.json`](data/videos.schema.json): 영상 메타데이터와 사용 조건 기록 계약.
- [`data/ground_truth.schema.json`](data/ground_truth.schema.json): 사람이 확인한 사실·근거·구간의 기록 계약.
- [`docs/results-format.md`](docs/results-format.md): 실행별 JSONL과 집계 CSV 계약.
- [`docs/selection.md`](docs/selection.md): 실험 뒤 선택 근거를 기록할 자리.
- [`src/README.md`](src/README.md): 후속 실행 코드의 책임 경계.

실제 API 키는 `GEMINI_API_KEY` 환경변수로만 전달합니다. 키를 코드, 설정 파일, `.env`, 로그, Git에 저장하지 않습니다. 공개 저장소에는 이용 허락을 확인한 메타데이터와 공유 가능한 평가 정보만 올립니다. 제한된 transcript와 원시 API 응답 및 로컬 결과는 `.gitignore` 대상입니다.

## 다음 단계

영상 3~5개의 이용 조건과 수동 ground truth를 확정한 뒤 실행기를 구현하고, 비용 상한을 정한 후 Pilot을 실행합니다. 현재는 아직 Benchmark를 실행할 수 없습니다.

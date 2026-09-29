# Pilot 평가 기준

Pilot 영상 4개는 유지하며 각 조건을 동일 횟수 반복합니다. 영상 언어(영어 3개, 한국어 1개)별 결과를 분리합니다. 접근 실패·요청 제한·파싱 실패·실행 불가도 결과에서 누락하지 않습니다. 사람 검수 ground truth가 `approved`가 되기 전에는 정답률·영상 근거 통과율을 확정하지 않습니다.

## Video Grounding Benchmark

Gemini 영상 분석과 **해당 영상에 사용할 권한이 확인된** transcript/caption을 별도 방식으로 기록합니다. transcript가 없거나 사용 권한이 확인되지 않으면 해당 조건은 `not_run`입니다. 모델이 만든 텍스트를 원본 transcript로 표시하지 않습니다.

| 항목 | 자동 기록 | 사람 검수 |
| --- | --- | --- |
| 입력·API | 영상 접근, transcript 권한/가용성, 오류 범주 | 입력 권한 및 출처 확인 |
| 사실·근거 | 추출 사실 수, 발화/화면 표시, 제시된 구간 | `approved` ground truth 및 실제 영상과 대조해 사실 존재·출처 판정 |
| Timestamp | 값의 존재·형식·범위 | 실제 영상 해당 장면/발화와 대조해 정확성 판정 |
| 누락·환각 | 자동으로 사실성 판정하지 않음 | ground truth 대비 중요한 사실 누락, 영상에 없는 주장 기록 |
| 사용량 | 지연, input/output/thinking tokens, 제공된 usage, 비용 | 가격·권한 적용 조건 확인 |

근거가 영상에 있더라도 timestamp가 틀릴 수 있으며, 정확한 timestamp라도 모델의 해석이 틀릴 수 있습니다. transcript는 발화 근거의 비교 기준이지만 화면에만 나온 정보까지 포함한다고 가정하지 않습니다.

## Quiz Generation Benchmark

영상별로 **한 번 고정한 동일 `contentText`와 해시**, 같은 질문 3개·보기 4개·프롬프트 버전·출력 계약을 `gemini-3.8-flash`와 `deepseek-flash`에 제공합니다. 모델별 입력 텍스트가 달라지면 생성 모델 비교로 집계하지 않습니다. 출력은 같은 JSON/Parser/Validator 계약으로 평가합니다.

| 항목 | 자동 기록 | 사람 검수 |
| --- | --- | --- |
| API/파싱/검증 | 상태, 오류 범주, 필수 필드·타입, 문제·보기 수, 0-based 정답 인덱스 | 필요 시 오류 분류 확인 |
| BE 호환성 | `sourceEvidence ∈ contentText` 및 Core 계약 검사 | 문자열 근거의 의미 확인 |
| 정답 정확성·유일성 | 자동으로 사실성 판정하지 않음 | 고정 입력·실제 영상 및 ground truth와 대조, 정답 하나인지 확인 |
| 근거성·환각 | 문자열 포함 여부만 자동 판정 | `sourceEvidence`가 **정답을 실제로 뒷받침**하는지, 영상 밖 주장은 없는지 확인 |
| 한국어·모호성 | 중복 문구 후보 탐지 | 질문·보기의 자연스러움, 중의성 및 여러 정답 가능성 확인 |
| 안정성 | 같은 조건 반복 결과와 실패율 | 품질 변동 확인 |
| 지연·사용량·비용 | latency, input/output/thinking tokens, 제공된 usage, 공식 가격 기준 추정 | 가격 적용 조건 검토 |

## End-to-End Benchmark

Gemini Grounding → Gemini Quiz, Gemini Grounding → DeepSeek Quiz, Gemini 직접 Quiz, 권한 있는 transcript → 두 Quiz 모델을 각각 실행합니다. Grounding과 Quiz 결과를 별도 실행 ID로 보존해 실패 계층, 전체 지연·비용, 최종 퀴즈 품질을 함께 기록합니다. 직접 Quiz는 `contentText`가 없으므로 `beCompatibility = not_applicable`이며 BE 검증 통과율의 분모에도 넣지 않습니다.

사람 검수는 사실·질문별로 `pass`, `fail`, `uncertain`을 기록하고 판단 근거를 남깁니다. `uncertain`을 성공으로 합산하지 않습니다. 승인되지 않은 ground truth 후보로는 확정 품질 점수를 만들지 않습니다. Timestamp나 모델이 작성한 `sourceEvidence` 문장만으로 영상 사실성이 입증되지 않습니다. 권한 있는 transcript가 없으면 해당 조건을 `not_run`으로 기록하고 다른 방식의 성공으로 대체하지 않습니다.

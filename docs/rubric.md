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

`omission`과 `hallucination`은 `contentText`와 `groundingFacts` 전체를 함께 살펴 사람이 판정합니다. Ground truth는 영상의 모든 사실 목록이 아니라 사전에 사람이 선정·승인한 핵심 사실 집합입니다. 단순 문자열·키워드 일치로 자동 판정하지 않습니다.

- `omission`: 승인된 ground truth의 각 핵심 의미가 `contentText` 또는 `groundingFacts` 중 하나 이상에 충분히 반영되면 `pass`, 하나 이상의 핵심 의미가 양쪽 모두에서 빠졌으면 `fail`입니다. 검토했지만 의미 포함 여부를 신뢰성 있게 결정하기 어려우면 `uncertain`, 아직 검토하지 않았으면 `null`입니다.
- `hallucination`: `contentText`와 `groundingFacts`의 실질적인 주장을 원본 영상과 대조합니다. 모두 영상에서 확인되면 `pass`, 영상에서 확인되지 않는 주장이 하나 이상 있으면 `fail`입니다. 검토했지만 영상 근거 여부를 신뢰성 있게 결정하기 어려우면 `uncertain`, 아직 검토하지 않았으면 `null`입니다. Ground truth에 없는 주장도 영상에서 확인되면 그 이유만으로 환각이 아닙니다.

### Pilot v2 Grounding 사람 검수 (`video-grounding-v2`·`video-grounding-v3`)

Pilot v2(`pilot-v2`)는 pilot-v1의 단순 반복이 아니라 개선된 조건의 별도 실험입니다. Grounding prompt `video-grounding-v2`는 v1의 facts 규칙에 다음 `contentText` 계약을 더합니다: (1) 영상의 핵심 정보를 충분히 포함, (2) 영상에 근거하지 않은 사실 금지, (3) 숫자·고유명사·인과관계 등 구체적인 정보 보존, (4) 서로 다른 Quiz 문제를 만들 수 있는 서로 구별되는 정보 보존, (5) 한국어, (6) 같은 Grounding의 facts와 모순 금지, (7) 메타 요약 문장 금지. 고정 최소 길이는 두지 않습니다. A/B Quiz의 공통 입력은 계속 승인된 `contentText`뿐이며 facts는 Quiz 입력에 넣지 않습니다. 현재 Pilot v2 설정의 `video-grounding-v3`는 같은 규칙에 timestamp 표현만 바꿉니다. Provider는 `"MM:SS"` 문자열을 쓰고 실행기가 경과 초로 변환해 저장합니다([결과 형식](results-format.md)). 검수 항목과 판정 규칙은 v2와 같습니다.

v2 Grounding은 사람이 아래 다섯 항목을 각각 `pass`, `fail`, `uncertain`으로 판정해야 Quiz에 쓸 수 있습니다. 판정은 사람이 입력하며 코드가 만들지 않습니다.

| 항목 | 판정 내용 |
| --- | --- |
| `factualAccuracy` | `contentText`의 주장이 원본 영상과 일치하는가 |
| `keyInformationCoverage` | 영상의 핵심 정보가 빠지지 않았는가 |
| `factsConsistency` | 같은 Grounding의 facts와 모순이 없는가 |
| `koreanConsistency` | 한국어로 일관되게 작성됐는가 |
| `contentTextContractCompliance` | 위 `contentText` 계약(구체성, 서로 구별되는 정보, 메타 요약 금지 등)을 지키는가 |

다섯 항목이 모두 `pass`면 `approved`, 하나라도 `fail` 또는 `uncertain`이면 `rejected`입니다. 이 전체 상태만 판정에서 결정적으로 유도합니다. `rejected`는 품질 판정이며 technical failure가 아니므로 재시도 대상이 아닙니다. rejected Grounding으로는 A/B Quiz를 실행하지 않습니다. 후보 선택 편향을 막기 위해 v2 Grounding 조건(`videoId`·`method`·`model`·`repetition`·Grounding `promptVersion`)에는 성공한 Grounding 후보를 하나만 둡니다. 성공이 하나라도 생기면 검수 결과와 관계없이 그 조건은 다시 생성하지 않으며, technical failure 뒤에만 다음 attempt를 실행합니다. 재생성이나 여러 후보 중 선택이 필요하면 별도의 사람 결정으로 다룹니다. 한 Pilot v2 실험의 결과는 하나의 결과 디렉터리에서 관리합니다. 기존 `omission`·`hallucination`과 fact별 판정은 그대로 별도 기록입니다. pilot-v1과 legacy Grounding에는 이 검수를 요구하지 않습니다.

## Quiz Generation Benchmark

Pilot v2 Quiz A/B·Direct C는 정확히 3문항을 요청하되 schema 배열 길이로 강제하지 않습니다. 실제 `questionCount != 3`은 API 성공 이후의 reliability/계약 실패(`validatorStatus=fail`)로 기록합니다. parsing·계약·사람 품질 판정이 실패해도 해당 조건에 `apiStatus=success`가 있으면 다시 생성하지 않습니다. technical `apiStatus=error`만 다음 live attempt를 허용합니다. 공식 v2 Quiz·Direct live는 내부 HTTP retry 없이(`retry_attempts=0`) 실행합니다. Grounding technical retry는 Issue #23에서 변경하지 않습니다. Provider별 thinking/reasoning medium은 동일한 추론량을 뜻하지 않습니다. pilot-v1의 평가와 실행 정책은 유지합니다.

**한 번 고정한 동일 `contentText`와 해시**(Pilot v1은 영상·prompt 버전별, Pilot v2는 영상·prompt 버전·repetition별로 같은 repetition의 승인 Grounding에서 고정하며 repetition마다 다를 수 있음), 같은 질문 3개·보기 4개·프롬프트 버전·출력 계약을 `gemini-3.8-flash`와 `gpt-5.4-mini`에 제공합니다. 모델별 입력 텍스트가 달라지면 생성 모델 비교로 집계하지 않습니다. 출력은 같은 JSON/Parser/Validator 계약으로 평가합니다.

| 항목 | 자동 기록 | 사람 검수 |
| --- | --- | --- |
| API/파싱/검증 | 상태, 오류 범주, 필수 필드·타입, 문제·보기 수, 0-based 정답 인덱스 | 필요 시 오류 분류 확인 |
| Benchmark 내부 Quiz 형식/계약 검사 | 구조·개수·인덱스·프롬프트 버전과 `sourceEvidence ∈ contentText` 문자열 포함 검사 | 문자열 근거의 의미 확인 |
| 정답 정확성·유일성 | 자동으로 사실성 판정하지 않음 | 고정 입력·실제 영상 및 ground truth와 대조, 정답 하나인지 확인 |
| 근거성·환각 | 문자열 포함 여부만 자동 판정 | `sourceEvidence`가 **정답을 실제로 뒷받침**하는지, 영상 밖 주장은 없는지 확인 |
| 한국어·모호성 | 중복 문구 후보 탐지 | 질문·보기의 자연스러움, 중의성 및 여러 정답 가능성 확인 |
| 안정성 | 같은 조건 반복 결과와 실패율 | 품질 변동 확인 |
| 지연·사용량·비용 | latency, input/output/thinking tokens, 제공된 usage, 공식 가격 기준 추정 | 가격 적용 조건 검토 |

### Pilot v2 Human Quiz 블라인드 평가

Pilot v2의 Quiz A(Gemini Grounding 승인 contentText → Gemini Quiz), B(같은 contentText → OpenAI Quiz), C(YouTube → Gemini Direct Quiz)를 하나의 블라인드 세션에서 같은 기준으로 평가합니다. 생성 과정이 아니라 최종 Quiz Set이 원본 영상 기준으로 얼마나 좋은지를 봅니다. 평가자는 원본 YouTube 영상과 Quiz Set만 보며, A/B에도 contentText를 보여주지 않습니다. 모든 판정은 `pass`, `fail`, `uncertain` 중 하나이며 제출할 때 비워 둘 수 없습니다.

문항별 판정은 기존 6개 항목(`answerAccuracy`, `uniqueAnswer`, `evidenceSupportsAnswer`, `videoGrounding`, `koreanQuality`, `hallucination`)과 `reviewNote`를 그대로 사용합니다. Quiz Set 전체에는 다음 3개 항목과 set-level `reviewNote`를 판정합니다.

| 항목 | pass | fail | uncertain |
| --- | --- | --- | --- |
| `coverage` | 3문항이 영상의 서로 다른 핵심 내용을 적절히 다루고 중요한 핵심을 명백히 놓치지 않음 | 한 부분·세부사항에 편중되거나 영상의 명확한 핵심 내용이 빠짐 | 영상만으로 신뢰성 있게 판단하기 어려움 |
| `redundancy` | 사실상 같은 사실·정답을 묻는 문항 중복이 없음 | 2개 이상의 문항이 사실상 같은 내용을 묻거나, 한 문제의 정답이 다른 문제의 답을 사실상 노출함 | 겹침이 실질적인 중복인지 판단하기 어려움 |
| `learningValue` | 3문항 전체가 영상의 핵심 내용 이해를 확인하는 학습용 Quiz Set으로 유용함 | 사소한 정보·표현 위주이거나, 영상 없이도 답할 수 있거나, 결함 문항 때문에 학습 확인용 세트로 기능하지 못함 | 신뢰성 있게 판단하기 어려움 |

정상 평가 대상은 `apiStatus=success`, `parseStatus=pass`, `questionCount=3`이고 모든 문항을 정상적으로 표시할 수 있는 Quiz Set입니다. validator 실패(예: `evidence_not_in_content`)만으로는 제외하지 않습니다. 그 밖의 조건은 평가 제외 사유로 기록합니다. `notRun`은 해당 조건에 기록된 실행 시도가 하나도 없다는 뜻이며 Quiz 모델 실패가 아닙니다. Grounding이 rejected되어 A/B Quiz를 의도적으로 실행하지 않은 경우도 여기에 해당합니다. `noApiSuccess`는 실행 시도는 있었지만 `apiStatus=success`가 하나도 없었다는 뜻(technical/API 실패)입니다. 그 밖에 `parseFailed`, `questionCountMismatch`, `invalidQuestionStructure`가 있습니다. `notRun`과 `noApiSuccess`를 같은 신뢰성 실패로 합산하지 않습니다. 평가 제외는 사람 평균에서 빠진다는 뜻일 뿐이며, 생성·검증 실패 자체는 Pilot 결과에 그대로 남아 모델 신뢰성 결과로 함께 보고해야 합니다. Human 세션은 원칙적으로 Pilot v2 생성 절차를 마친 뒤 만듭니다. 세션 생성은 모든 조건의 실행 완료를 요구하지 않으므로, 생성 도중에 만든 세션에서는 아직 실행하지 않은 조건이 `notRun`으로 기록됩니다.

알려진 블라인드 한계: `evidenceSupportsAnswer` 판정을 위해 `sourceEvidence`를 보여주므로, A/B는 같은 contentText를, C는 영상을 인용하는 문체 차이로 조건을 간접 추측할 가능성이 일부 있습니다. 조건, model, provider, method, runId 같은 직접적인 조건 metadata는 내보내지 않습니다.

## End-to-End Benchmark

Gemini Grounding → Gemini Quiz, Gemini Grounding → OpenAI Quiz, Gemini 직접 Quiz, 권한 있는 transcript → 두 Quiz 모델을 각각 실행합니다. Grounding과 Quiz 결과를 별도 실행 ID로 보존해 실패 계층, 전체 지연·비용, 최종 퀴즈 품질을 함께 기록합니다. `beCompatibility=pass`와 `beCompatibilityRate`는 Benchmark 내부 로컬 검사 통과(율)이며 실제 Cking-BE validator 실행 결과가 아닙니다. Grounding에는 이 필드를 적용하지 않습니다. 직접 Quiz는 고정 `contentText`가 없어 동일한 로컬 evidence/contentText 검사를 할 수 없으므로 `beCompatibility=not_applicable`이며 해당 통과율의 분모에 넣지 않습니다. 이는 직접 Quiz의 품질·정답 정확성·실제 BE 호환성 통과를 뜻하지 않습니다.

사람 검수는 사실·질문별로 `pass`, `fail`, `uncertain`을 기록하고 판단 근거를 남깁니다. `uncertain`을 성공으로 합산하지 않습니다. 승인되지 않은 ground truth 후보로는 확정 품질 점수를 만들지 않습니다. Timestamp나 모델이 작성한 `sourceEvidence` 문장만으로 영상 사실성이 입증되지 않습니다. 권한 있는 transcript가 없으면 해당 조건을 `not_run`으로 기록하고 다른 방식의 성공으로 대체하지 않습니다.

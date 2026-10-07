# LLM-as-a-Judge 보조 평가 프로토콜

Pilot repetition 1의 A/B Quiz 결과에 대한 사후(post-hoc) 보조 평가입니다. Human Evaluation이 최종 기준(reference)이며, Judge 결과로 Human Evaluation, ground truth, 기존 Pilot 결과를 수정하지 않습니다. Judge는 저장된 결과를 읽기만 합니다.

**실제 Provider를 호출하는 Judge 실행(`--live`)은 별도 승인 후에만 진행합니다.** 실행 전에 실행 시점의 공식 문서로 model ID, 요청 필드, usage·캐시 필드, retry hint, 가격을 다시 확인하고, 출력 상한·timeout·비용 상한을 정합니다.

## 범위

- 대상: A(승인된 고정 `contentText` → `gemini-3.8-flash` Quiz)와 B(같은 `contentText` → `gpt-5.4-mini` Quiz).
- Direct Quiz(C)는 범위 밖입니다. C는 공통 승인 `contentText`를 생성 입력으로 쓰지 않았으므로 같은 텍스트 기준 rubric을 적용할 수 없습니다. C에 A/B의 `contentText`를 사후 참고 자료로 주지 않으며, C의 품질은 기존 Human Evaluation(실제 영상 기준)으로 보고합니다.
- Judge는 영상이 아니라 승인된 `contentText`만을 근거로 판정합니다.

## 대상 선정

- 원천 무결성(전제 조건): 승인된 Grounding 연결과 승인 추적 필드, `contentText` SHA-256 일치, A/B pair 동일성(`videoId`, `repetition`, `sourceGroundingRunId`, `contentTextSha256`, `promptVersion`), 조건별 성공 Quiz가 하나인지 확인합니다. 하나라도 어긋나면 계획 단계에서 중단하고 Judge를 호출하지 않습니다.
- Pointwise: `apiStatus=success`, `parseStatus=pass`인 Quiz의 문항 중 최소 구조(비어 있지 않은 `question`·`explanation`·`sourceEvidence`, 비어 있지 않은 문자열 `options`, bool이 아닌 정수이며 범위 안의 `correctOptionIndex`)를 만족하는 문항. validator 계약(보기 4개, 중복, evidence 문자열 포함, promptVersion)은 제외 조건이 아닙니다. 구조를 만족하지 못한 문항만 제외하고 원래 `questionIndex`를 유지합니다.
- Pairwise: 양쪽 모든 문항이 최소 구조를 만족하고(제외 문항 없음) 평가 가능한 문항 수가 같은 pair만 대상입니다. 일부 문항끼리 비교하지 않습니다. 대상에서 빠진 pair는 Judge를 호출하지 않고 제외 기록을 남기며, 이를 선호 결과로 해석하지 않습니다.
- 현재 Pilot 기준 기대값: Pointwise 8 set, 22문항(Mars B는 기존 1문항), Pairwise 3 pair(Water, Methane, KARI). Mars pair는 문항 수 불일치(3 대 1)로 제외됩니다. 확인은 `python -m src.judge_runner plan`(Provider 호출 없음, 파일 쓰기 없음)으로 합니다.

## Rubric

모든 항목은 `contentText` 기준입니다.

**Pointwise** (문항 × 항목마다 `pass | fail | uncertain` + reason, 점수나 set 종합 점수 없음)

| 항목 | 의미 |
| --- | --- |
| `textAnswerCorrect` | 정답이 `contentText` 기준으로 맞는가 |
| `uniqueAnswer` | 정답으로 성립하는 보기가 하나뿐인가. 둘 이상의 보기가 실제 정답일 때만 fail이며, 표현만 모호하면 `questionClarity`에서 평가 |
| `evidenceSupportsAnswer` | `sourceEvidence`가 정답을 실제로 뒷받침하는가 |
| `questionClarity` | 질문과 보기가 명확하고 모호하지 않은가 |
| `koreanQuality` | 한국어가 자연스럽고 문법적으로 적절한가 |
| `distractorQuality` | 오답 보기가 무의미하거나 지나치게 쉽지 않은가 |
| `textFaithfulness` | 질문·정답 보기·설명·evidence가 `contentText` 밖의 사실을 근거 없이 주장하지 않는가(오답 보기 제외) |

두 Judge의 Pointwise 결과를 합쳐 새 verdict를 만들지 않습니다.

**Pairwise** (중립 라벨 `SET_1`/`SET_2`, 저장 시 canonical A/B로 변환)

- validity: set별 결함 문항 `{questionIndex, defectTypes, reason}`. `defectTypes`는 `answerIncorrect`, `notUnique`(두 개 이상의 보기가 실제 정답), `evidenceUnsupported`, `unfaithful`(`textFaithfulness`와 같은 범위)입니다.
- quality: `clarity`, `koreanQuality`, `distractorQuality`, `coverage`마다 `SET_1 | SET_2 | TIE`. 명확하고 실질적인 차이가 없으면 TIE입니다. `coverage`는 서로 다른 핵심 정보를 넓고 중복 없이 다루는 정도이며 문항 수는 근거가 아닙니다.
- call별 최종 결과(코드 계산): 결함 **문항** 수가 적은 쪽(`decidedBy=validity`, 한 문항의 여러 결함은 1개). 같으면 quality에서 이긴 항목 수(`decidedBy=quality`), 같으면 TIE(`decidedBy=tie`). quality는 validity를 뒤집지 않으며 전체 rubric 다수결은 쓰지 않습니다.
- 순서 합치기(Judge별, AB/BA): 같은 값이면 그 값(`orderAgreement=full`), 한쪽 선호와 TIE는 TIE(`partial`), A와 B는 `INCONSISTENT`(`contradictory`), 한 순서라도 최종 실패면 `INCOMPLETE`.
- Judge 합치기(pair, 대칭): 하나라도 `INCOMPLETE`면 `INCOMPLETE`, 그 외 하나라도 `INCONSISTENT`면 `INCONSISTENT`, 같으면 그 값(`full`), 선호와 TIE는 TIE(`partial`), A와 B는 `JUDGE_DISAGREEMENT`(`contradictory`). Judge 하나만으로 pair 결과를 정하지 않습니다.
- 상태와 선호는 별도 필드입니다. 결정된 결과는 `status=RESOLVED`와 `preference=A|B|TIE`, 나머지 상태는 `preference=null`입니다.

## Judge 모델과 입력

- `gpt-6.1-sol`(Responses API, strict structured output, `reasoning.effort=medium`)과 `gemini-3.8-flash`(Interactions API, JSON schema 응답, `thinking_level=medium`). 두 모델 모두 기본값에 의존하지 않고 `medium`을 명시합니다. Standard tier를 사용합니다. model ID와 문서 확인 날짜는 `configs/judge.yaml`과 실행 manifest에 남습니다.
- Judge 입력에는 `contentText`와 문항 필드(`question`, `options`, `correctOptionIndex`, `explanation`, `sourceEvidence`)만 들어갑니다. 생성 provider·모델, A/B 조건의 의미, `promptVersion`, `pricingReference`, validator 결과, 기대 문항 수, Human Evaluation과 reviewNote는 넣지 않습니다.
- Quiz 텍스트는 수정하거나 정규화하지 않습니다. 따라서 문체로 생성 계열을 추정할 가능성은 통제하지 못합니다.

## 해석상 한계

- Human은 실제 영상과 ground truth, Judge는 승인된 `contentText`를 기준으로 판정합니다. 이름이 같은 항목도 기준이 다릅니다.
- 서로 다른 계열의 Judge를 쓰는 것은, 잠재적인 자기 계열 선호가 있다면 서로 다른 방향으로 나타날 수 있도록 하기 위함입니다. 실제로 편향이 있는지, 상쇄되는지, 어느 Judge가 더 객관적인지는 이 Pilot으로 결론 내리지 않습니다.
- Gemini Judge는 A를 생성한 모델과 같은 `gemini-3.8-flash`입니다. 계열 효과와 Judge 모델 자체의 차이를 분리할 수 없습니다.
- Pairwise 대상은 3 pair, Pointwise는 22문항입니다. 결과는 개수로만 보고하고 비율, kappa, 신뢰구간, 유의성 검정, provider 우열 점수를 만들지 않습니다. 문항들은 독립 표본이 아닙니다.

## Human Evaluation과의 비교

- Pointwise: `textAnswerCorrect`↔`answerAccuracy`, `uniqueAnswer`↔`uniqueAnswer`, `evidenceSupportsAnswer`↔`evidenceSupportsAnswer`(기준 차이 명시), `koreanQuality`↔`koreanQuality`(직접), `textFaithfulness`↔`hallucination`(참고용). `questionClarity`, `distractorQuality`, Human `videoGrounding`은 일치 계산 대상이 아닙니다. 어느 한쪽이라도 `uncertain`이면 `NOT_COMPARABLE_UNCERTAIN`이며 Judge 간 일치에도 같은 규칙을 씁니다.
- Pairwise와 Human은 정답·예측 관계가 아닙니다. Human에는 A/B 직접 선호 판정이 없습니다.
- 탐색용 Human 유래 validity 비교: Human `answerAccuracy`, `uniqueAnswer`, `evidenceSupportsAnswer` 중 하나라도 fail인 문항 수를 비교해 `A | B | NO_VALIDITY_DIFFERENCE`, 세 필드에 `uncertain`이 있으면 `UNDETERMINED`입니다. 이것은 Human 판정이 아니라 규칙으로 계산한 값입니다. Judge 결과를 구성한 call이 모두 `decidedBy=validity`일 때만 비교하고, 그 외는 `NOT_COMPARABLE_DECISION_BASIS`, `JUDGE_DISAGREEMENT`·`INCONSISTENT`·`INCOMPLETE`는 `NOT_COMPARABLE_STATUS`입니다. Mars는 제외합니다.

## 재시도와 실패

- 논리적 측정(Judge × set 또는 Judge × pair × 순서)당 최대 3 technical attempts. attempt 안에서 429와 5xx는 HTTP 재시도 최대 1회입니다.
- 429/5xx 뒤에는 즉시 다시 보내지 않고 `Retry-After`를 따릅니다. 공식 hint가 없을 때의 대기 방식은 아직 정하지 않았으므로, 현재 구현은 이 경우 실행을 `retryWaitUnresolved`로 중단합니다(임시 fail-closed 상태이며 최종 프로토콜 결정이 아님).
- 다음 attempt 대상: timeout, network, HTTP 재시도 소진, `incomplete`, `refusal`, `parse_error`, `schema_error`, `semantic_error`. `incomplete` 응답에 모델 출력 텍스트가 있으면 진단용으로만 보존하며 결과로 쓰지 않습니다. 검증은 구조만 확인하며 판정 내용 때문에 출력을 거부하지 않습니다. 일부만 담긴 출력은 실패입니다.
- 실행 전체 중단: `client_error`(4xx), `api_key_missing`, `live_guard`(호출 수·비용 상한), 같은 측정의 3 attempts가 모두 `incomplete`인 경우(`repeatedIncomplete`). 실패 유형이 섞여 소진되면 그 측정만 `INCOMPLETE`로 두고 계속합니다.
- 재시도는 같은 입력(같은 `inputHash`)으로만 합니다. 첫 번째 유효 결과만 쓰고 성공한 측정은 다시 실행하지 않습니다. 성공이 여러 개면 가장 이른 attempt를 쓰고 나머지는 표시만 합니다.
- 측정 식별에 영향을 주는 설정(출력 상한, prompt·rubric·schema, 모델·추론 설정)이 바뀌면 새 실행으로 대상 전체를 다시 측정합니다. 같은 실행의 재개(`--run-id`)는 실행 모드(`fixture`/`live`), `configHash`, 원천 스냅샷, 계획된 측정이 모두 같을 때만 허용됩니다. fixture 실행과 live 실행은 서로 재개할 수 없습니다. HTTP 요청 상한과 비용 상한은 프로세스가 아니라 Judge 실행 전체에 적용됩니다. live 재개는 manifest에 기록된 운영 설정(요청·비용 상한, timeout, 입력 token 추정값, 출력 상한, 단가·출처·확인일)이 정확히 같아야 하며, `attempts.jsonl`의 모든 HTTP 요청 기록(실패·재시도 포함)에서 요청 수와 지출(비용을 아는 요청은 정산 비용, 모르는 요청은 사전 예약 비용)을 복원한 뒤 남은 한도로만 이어갑니다. 한도가 이미 소진됐거나, 요청 기록에 예약 비용이 없어 복원할 수 없는 실행은 Provider 호출과 상태 변경 전에 재개를 거부합니다. 이전 실행은 보존하되 새 실행 분석에 섞지 않습니다.

## 저장 구조

Judge 결과는 `results/judge/<judgeRunId>/`에만 저장합니다. `results/raw`, `results/evaluation`에는 쓰지 않으므로 기존 Pilot 저장 무결성 검사와 충돌하지 않습니다.

| 파일 | 내용 |
| --- | --- |
| `run.json` | 실행 manifest: 상태와 중단 사유, `configHash`와 식별 설정, 모델 문서 확인 날짜, 운영 설정(guard, 단가·출처·확인일), 원천 결과 경로와 파일 SHA-256 스냅샷 |
| `eligibility.jsonl` | 대상 선정과 제외 기록(문항 index, 사유, 문항 수). `contentText`와 Human 값은 저장하지 않음 |
| `measurements.jsonl` | 계획된 논리적 측정과 `inputHash` |
| `attempts.jsonl` | 모든 technical attempt(append-only): 상태와 `errorCategory`, HTTP 요청 기록, 요청·응답 모델, 추론 설정, usage, 추정 비용, 가격 출처 |
| `outputs/<attemptId>.json` | Judge 출력 원문과 검증된 결과 |
| `derived/` | 측정 상태, Pointwise verdict, Pairwise call·Judge·pair 결과, Human 비교, `report-counts.json`. 원본에서 `python -m src.judge_runner derive --run-id <id>`로 다시 생성 |

비용은 선택된 측정 기준(`selectedMeasurements`)과 실패 attempt를 포함한 총지출(`allAttempts`)로 나눠 보고합니다. 각각 `usd`(하나라도 비용을 모르면 `null`), `knownUsd`(알려진 비용 합계, 하한), `unknownCostCount`를 둡니다. usage가 없거나 형식이 잘못됐거나(음수, bool, 캐시 token이 입력 token보다 큼, `usage.invalidFields`에 기록) 계산 결과가 유한한 0 이상의 수가 아니면 비용은 `null`이며 0으로 바꾸지 않습니다. 이때 사전 예약한 추정 비용은 정산하지 않고 그대로 남겨 비용 상한 계산에 계속 반영합니다. 캐시 입력 단가는 생략하거나 유한한 0 이상의 수여야 합니다. Gemini Interactions API는 `usage.total_cached_tokens`(prompt 중 캐시된 부분)를 보고하지만, 이 값이 `total_input_tokens`에 포함되는지와 implicit 캐시에 캐시 단가가 어떻게 적용되는지는 공식 문서(2026-10-07 확인)에 명시되어 있지 않습니다. 따라서 캐시 token이 0보다 큰 Gemini 호출의 비용은 `null`로 두고 예약을 유지합니다. 정확한 계산식은 공식 문서로 의미가 확인된 뒤 적용합니다. HTTP 요청 기록에는 요청별 사전 예약 비용(`reservedUsd`)과 비용을 알게 된 경우의 정산 비용(`settledUsd`)이 남습니다. HTTP 요청 기록은 실제로 보낸 요청마다 남으며 요청 본문과 credential은 저장하지 않습니다.

## 실행

```bash
python -m src.judge_runner plan
```

```bash
python -m src.judge_runner run --fixture <fixture.json> --max-output-tokens-pointwise <N> --max-output-tokens-pairwise <N>
```

`--live` 실행은 별도 승인 후, 출력 상한, 실행당 HTTP 요청 상한, call당·전체 비용 상한, timeout, 종류별 입력 token 추정값, provider별 단가·출처·확인일을 모두 지정해야 합니다.

# 결과 형식 계약

실제 결과 파일은 저장소의 Git-ignore된 `results/` 안에 저장하며 Git에 올리지 않습니다. 다른 저장소 경로나 저장소 밖 경로는 실행기가 거부합니다. 원시 Provider 응답은 `results/raw/<runId>.json`에 별도로 두고, 공유할 때 권리·개인정보·비밀정보를 검토합니다. 각 JSONL 행은 [`run-result.schema.json`](run-result.schema.json)의 세 결과 유형 중 하나입니다. 현재 실행기는 fixture 전용이며 실제 Provider 호출은 지원하지 않습니다. fixture와 로컬 transcript는 `apiStatus=not_run`, API latency·token·cost는 `null`로 기록합니다. fixture가 모의한 오류는 `errorCategory=fixture_...`로 구분합니다.

## 분리된 실행 결과

- `results/video-grounding.jsonl`: `(videoId, method, repetition)`마다 한 행. `benchmarkType=video_grounding`, `method=gemini_video|authorized_transcript`. `runId`, `videoId`, `model`(transcript는 `null`), `apiStatus`, `latencyMs`, 토큰·비용, `groundingFacts`를 기록합니다. 사실별 `evidenceType`(발화/화면), 근거, 제시된 timestamp와 사람 판정 `factExists`, `evidenceTypeCorrect`, `timestampAccurate`를 분리합니다. 영상별 `omission`·`hallucination` 판정은 승인된 ground truth와 실제 영상을 확인한 뒤에만 기록합니다.
- `results/quiz-generation.jsonl`: `(videoId, contentTextSha256, model, promptVersion, repetition)`마다 한 행. `benchmarkType=quiz_generation`. 영상별 **동일한 고정 `contentText`**의 SHA-256을 두 모델에 공통으로 사용합니다. `sourceGroundingRunId`는 원천 Grounding 결과를 추적할 때만 사용하며, 입력 내용은 제한된 로컬 파일로 보관합니다. API·Parser·Validator 상태, BE 호환성, 질문별 정답 정확성·유일성·근거의 정답 지지·한국어 품질을 기록합니다.
- `results/end-to-end.jsonl`: `(videoId, method, repetition)`마다 한 행. `benchmarkType=end_to_end`. 두 단계 방식은 `groundingRunId`와 `quizRunId`를 참조합니다. 직접 Quiz는 두 ID가 모두 `null`이며 질문별 검수 결과를 이 행에 남깁니다. 단계별 실패와 총 latency·토큰·비용을 기록합니다. 직접 Quiz의 `beCompatibility`는 항상 `not_applicable`입니다. transcript 조건을 실행할 권한이나 원문이 없으면 `apiStatus=not_run`과 이유를 남깁니다.

공통 식별·호출 필드는 `runId`, `videoId`, `method`, `model`, `promptVersion`, `repetition`, `startedAt`, `apiStatus` (`success|error|not_run`), `errorCategory`, `latencyMs`, `inputTokens`, `outputTokens`, `thinkingTokens`, `estimatedCostUsd`, `pricingReference`입니다. transcript Grounding처럼 적용되지 않는 값과 Provider가 usage를 반환하지 않은 값은 `null`로 두며 `0`으로 채우지 않습니다. 로컬 원본의 경로·전체 텍스트·API Key는 공유 결과 행에 넣지 않습니다.

사람 판정 값은 `pass|fail|uncertain`입니다. 평가하지 않았거나 승인된 ground truth가 없어 확정할 수 없는 값은 `null`입니다. `uncertain`과 `null`을 성공으로 합산하지 않습니다. `evidenceTextContained`는 문자열 검사이고 `evidenceSupportsAnswer` 및 `videoGrounding`은 별도의 사람 판정입니다. timestamp의 값이 있다는 것과 `timestampAccurate=pass`는 다른 의미입니다.

## 집계 CSV

세 평가를 각각 집계합니다. 모든 집계에는 `benchmarkType,method,model,promptVersion,runs,apiSuccessRate,latencyMeanMs,latencyP95Ms,meanInputTokens,meanOutputTokens,meanThinkingTokens,meanEstimatedCostUsd`와 적용 가능한 품질 지표를 포함합니다.

- Grounding: `factExistsPassRate,evidenceTypePassRate,timestampAccuracyPassRate,omissionFailRate,hallucinationFailRate`
- Quiz: `parseSuccessRate,validatorSuccessRate,beCompatibilityRate,answerAccuracyPassRate,uniqueAnswerPassRate,evidenceSupportsAnswerPassRate,koreanQualityPassRate`
- End-to-end: `parseSuccessRate,beCompatibilityRate,videoGroundingPassRate,answerAccuracyPassRate,hallucinationFailRate,meanTotalCostUsd`

각 비율은 **평가 가능한 승인 ground truth/질문 수**를 분모로 기록하고, `uncertain`·`null` 개수와 실행 불가 건수를 별도로 표시합니다. `gemini_direct_quiz`의 `beCompatibilityRate`는 빈 값입니다. 모델 품질을 비교하는 Quiz 집계는 같은 `contentTextSha256`과 프롬프트 버전의 실행만 묶습니다.

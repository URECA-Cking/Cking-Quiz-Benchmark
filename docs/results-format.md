# 결과 형식 계약

실제 결과 파일은 로컬 `results/`에 저장하며 Git에 올리지 않습니다. 원시 Provider 응답은 별도 로컬 파일로 두고, 공유할 때 권리·개인정보·비밀정보를 검토합니다.

## 실행별 JSONL

한 줄이 `(videoId, method, model, promptVersion, repetition)` 한 실행입니다. 필드:

- 식별: `runId`, `videoId`, `method` (`A|B|C`), `model`, `promptVersion`, `repetition`, `startedAt`
- 호출: `apiStatus` (`success|error|not_run`), `errorCategory`, `latencyMs`, `inputTokens`, `outputTokens`, `thinkingTokens`, `estimatedCostUsd`, `pricingReference`
- 형식: `parseStatus`, `questionCount`, `optionCountValid`, `beCompatibility` (`pass|fail|not_applicable`), `evidenceTextContained`
- 사람 검수: 질문별 `relevance`, `answerAccuracy`, `videoGrounding`, `ambiguity`, `hallucination` (`pass|fail|uncertain`)과 `reviewNote`

누락된 usage와 비용은 `null`로 기록합니다. `0`으로 채우지 않습니다. B의 `beCompatibility`는 항상 `not_applicable`입니다. 모델이 만든 근거 설명의 문자열 일치와 영상 근거성은 별도 필드입니다.

## 집계 CSV

`method,model,promptVersion,runs,apiSuccessRate,parseSuccessRate,beCompatibilityRate,relevancePassRate,answerAccuracyPassRate,videoGroundingPassRate,hallucinationFailRate,latencyMeanMs,latencyP95Ms,meanInputTokens,meanOutputTokens,meanThinkingTokens,meanEstimatedCostUsd`를 사용합니다. 집계값의 분모와 `uncertain` 처리 기준을 함께 기록합니다. B의 `beCompatibilityRate`는 빈 값입니다.

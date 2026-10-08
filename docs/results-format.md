# 결과 형식 계약

## Pilot v2 Quiz·Direct 실행 조건 (Issue #23)

`pilot-v2`는 Gemini Quiz A·Direct C의 `generation_config.thinking_level=medium`, OpenAI Quiz B의 `reasoning.effort=medium`을 config와 실행 전 일치 검사로 통제합니다. Provider별 medium은 동일한 추론량을 뜻하지 않습니다. Grounding generation 설정은 유지하며 temperature/top_p 및 새 결과 필드는 추가하지 않습니다.

정확히 3문항을 요청하지만 요청 schema의 minItems/maxItems로 강제하지 않습니다. 문항 수가 다르면 `apiStatus=success`, `validatorStatus=fail`, `errorCategory=quiz_contract_error`로 보존합니다. 같은 v2 Quiz·Direct 조건에서 API success가 하나라도 있으면 parsing·validator·사람 품질 판정과 관계없이 terminal로 재생성을 거부합니다. technical `apiStatus=error` 뒤에만 다음 live attempt를 허용합니다. fixture/local의 `not_run`은 실제 API 성공이 아닙니다.

Quiz A/B 조건은 videoId·method·model·promptVersion·contentTextSha256·repetition, Direct C 조건은 videoId·method·model·promptVersion·repetition입니다. attempt와 terminal 검사에 같은 identity를 사용하며 Direct v1/v2는 독립된 sequence입니다. 기존 행은 migration하지 않습니다. 과거 two-stage End-to-End 행의 identity는 변경하지 않습니다.

v2 Quiz·Direct 공식 live는 `retry_attempts=0`만 허용합니다. Grounding technical retry는 Issue #23에서 변경하지 않습니다. CLI와 Router 직접 호출 모두 HTTP 전에 검사하며 명시한 thinking/reasoning CLI 값이 config와 다르면 거부합니다. v1의 prompt·generation·retry·품질 실패 후 실행 정책은 유지합니다. 검사는 기존 lock 안에서 해당 results-dir의 결과를 기준으로 수행합니다.

실제 결과 파일은 저장소의 Git-ignore된 `results/` 안에 저장하며 Git에 올리지 않습니다. 다른 저장소 경로나 저장소 밖 경로는 실행기가 거부합니다. `results/raw/<runId>.json`에는 완료 상태, Provider 이름, 숫자형 token usage만 기록하고 fixture는 식별 표지만 기록합니다. 자유형 모델 출력은 raw metadata에 넣지 않고 `results/evaluation/<runId>.json`에 normalized 평가 데이터로 별도 저장합니다. 평가 데이터에는 입력에서 반사된 민감정보가 있을 수 있으므로 공유 전 검토해야 합니다. 각 JSONL 행은 [`run-result.schema.json`](run-result.schema.json)의 세 결과 유형 중 하나입니다. 현재 실행기는 fixture 기본값이며 명시적인 live 옵션·상한을 제공한 한 조건에서만 Provider 호출을 허용합니다. fixture와 로컬 transcript는 `apiStatus=not_run`, API latency·token·cost는 `null`로 기록합니다. fixture가 모의한 오류는 `errorCategory=fixture_...`로 구분합니다. `--total-cost-limit`은 한 CLI 실행의 추정 비용 guard로, 실제 청구액이나 여러 실행에 걸친 누적 상한이 아닙니다.

Pilot 실행은 한 번에 하나씩 순차적으로 수행합니다. 같은 results 디렉터리에서는 결과를 바꾸는 작업(`run_grounding`, `run_quiz`, `run_end_to_end`, `approve_content`, `review_grounding`)을 한 번에 하나만 허용합니다. 두 번째 작업은 기다리지 않고 results 디렉터리의 `.pilot.lock` OS 잠금(Windows `msvcrt.locking`, POSIX `fcntl.flock`)을 얻는 단계에서 바로 거부되며, Provider 호출과 결과 기록을 하지 않습니다. 잠금은 작업이 끝나거나 프로세스가 종료되면 OS가 해제하므로 `.pilot.lock` 파일이 남아 있다는 것만으로 잠긴 상태는 아닙니다. 서로 다른 results 디렉터리는 서로 막지 않습니다. 이는 동시 실행을 지원한다는 뜻이 아니라 겹친 작업을 거부한다는 뜻이며, 네트워크 파일 시스템에서의 잠금 동작은 보장하지 않습니다. 저장 전 기존 JSONL과 필수 연결 파일을 검사하고 손상·부분 저장 흔적을 발견하면 자동 복구하거나 삭제하지 않고 다음 실행을 중단합니다. 개별 raw/evaluation JSON과 summary JSONL은 같은 디렉터리의 임시 파일을 완성한 뒤 교체하지만, 여러 파일을 하나의 원자적 transaction으로 저장하는 것은 아닙니다. API 응답 후 저장 완료 전 종료 시 실행 기록이 남지 않을 수도 있습니다.

## 분리된 실행 결과

- `results/video-grounding.jsonl`: `(videoId, method, model, repetition, attempt)` 실행마다 한 행. `benchmarkType=video_grounding`, `method=gemini_video|authorized_transcript`. `runId`, `videoId`, `model`(transcript는 `null`), `apiStatus`, `latencyMs`, 토큰·비용, `groundingFacts`를 기록합니다. 생성된 `contentText`의 UTF-8 SHA-256은 `contentTextSha256`에 기록하고, 사람 검수 전 `contentTextApprovalStatus`는 `null`로 둡니다. 사람이 해당 텍스트를 명시적으로 승인한 경우에만 Grounding 행의 승인 상태를 `approved`로 갱신합니다. 필드가 없는 기존 행은 미승인입니다. 사실별 `evidenceType`(발화/화면), 근거, 제시된 timestamp와 사람 판정 `factExists`, `evidenceTypeCorrect`, `timestampAccurate`를 분리합니다. 영상별 `omission`·`hallucination` 판정은 승인된 ground truth와 실제 영상을 확인한 뒤에만 기록합니다.
  - timestamp: `timestampStartSeconds`·`timestampEndSeconds`는 영상 시작부터의 경과 초입니다(예: 01:30.5 → 90.5). MM:SS 표기에서 콜론만 뺀 숫자가 아닙니다. 근거 위치를 특정할 수 없으면 `null`이며 한쪽만 `null`이어도 허용합니다. 저장되는 두 필드의 의미는 Grounding prompt version과 관계없이 같습니다.
  - Provider timestamp 표현: `video-grounding-v1`·`v2`는 Provider가 두 필드에 경과 초 숫자를 직접 씁니다. 실제 `video-grounding-v2` 호출에서 Provider가 이 지시를 지키지 않고 MM:SS의 콜론만 뺀 숫자(01:41.2 → 141.2)를 반환했고, 숫자 하나만으로는 이 오류를 판별할 수 없습니다. 그래서 `video-grounding-v3`는 Provider에게 `timestampStart`·`timestampEnd`를 `"MM:SS"` 또는 `"MM:SS.s"` 문자열(분 1~3자리, 초 00~59, 소수 최대 3자리, `null` 허용)로 받고, 실행기가 `분 × 60 + 초`로 결정적으로 변환해 위 두 필드에 경과 초로 저장합니다(01:41.2 → 101.2). 성공한 행의 evaluation 파일 facts도 같은 경과 초 필드로 저장하며 Provider 문자열은 저장 계약에 남기지 않습니다. 실패 행은 기존처럼 Provider 출력을 진단용 evaluation으로 남길 수 있습니다. 형식이 맞지 않는 값(초 60 이상, 부호, 숫자 아닌 문자, 구분자 누락, 시간 단위 포함, 숫자·bool 타입, 필드 누락)은 보정하지 않고 `invalid_grounding_response`로 실패 처리합니다. Gemini의 문서화된 schema 범위에 문자열 `pattern`이 없어 schema는 `string | null` 타입과 설명만 지정하고, 정확한 형식은 실행기가 검사합니다.
  - timestamp 검사: 새 Provider 응답에서 각 값(`video-grounding-v3`은 변환된 경과 초)은 `null`이거나 bool이 아닌 유한한 숫자(NaN·Infinity 불가)로 0 이상이어야 하고, `data/videos.jsonl`에 `durationSeconds`가 있으면 존재하는 start와 end 각각이 `durationSeconds` 이하여야 하며(tolerance 없음), 두 값이 모두 있으면 `start <= end`여야 합니다. 위반 시 값을 보정하지 않고 `invalid_grounding_response`로 실패 처리합니다. 이때 Provider 출력이 표준 JSON으로 저장될 수 없으면(NaN·Infinity 포함) 오류 행과 raw metadata만 저장하고 evaluation 파일은 남기지 않습니다. `durationSeconds`가 없는 영상은 길이 검사를 하지 않습니다. 기록된 `durationSeconds`가 있으면 AI Grounding은 Provider 호출 전에 그 값이 bool이 아닌 양수(int/float)인지 확인하고, 아니면 결과를 저장하지 않고 실행을 거부합니다. `durationSeconds`는 정수입니다. 현재 네 영상 모두 출처는 YouTube 영상 페이지 메타데이터의 `approxDurationMs`이며, 그 값을 초 단위로 올림해 기록했습니다(페이지의 `duration` 표기와 같음). 기록된 정수는 strict 상한으로 쓰입니다. 올림을 쓰면 `approxDurationMs`를 내림·절삭한 값보다 정상 timestamp를 상한 때문에 잘못 거부할 위험이 줄고, `approxDurationMs`보다 1초 미만 큰 값까지는 통과할 수 있습니다. `approxDurationMs`는 근사 메타데이터이며 실제 재생 길이와의 정확한 관계는 저장소 근거로 확인되지 않았습니다. 따라서 이 상한은 실제 재생 길이를 보장하지 않는 추가 방어이고, 실제 길이와 메타데이터 근사치의 차이로 인한 오거부·오수용 가능성을 완전히 없애지 않습니다. 또한 길이 상한은 상한을 넘는 값만 잡으며, 영상 길이 안에 들어오는 MM:SS형 숫자는 구분하지 못합니다. 이 검사는 저장된 기존 결과에 소급 적용하지 않습니다.
  - Grounding `promptVersion`: AI Grounding(`gemini_video`) 행은 `pilot.video_grounding.prompt_version`(`configs/pilot.yaml`은 `video-grounding-v1`, `configs/pilot-v2.yaml`은 `video-grounding-v3`)을 기록하며, 설정이 없거나 빈 값이면 Provider 호출 전에 실행을 거부합니다. `authorized_transcript` 행은 AI prompt를 사용하지 않으므로 `null`입니다. 이 값 도입 전 Grounding 행의 `null`은 Grounding prompt versioning 도입 전 결과라는 뜻입니다. 전역 `prompt_version`(`pilot-v1`)은 Quiz·Direct Quiz에만 쓰입니다. Grounding attempt 번호는 아래 조건에 Grounding `promptVersion`을 더해 매기며, versioning 도입 전 `null` 행은 attempt 번호와 offline 집계 history에서 `video-grounding-v1`과 같은 그룹으로 셉니다. 그래서 pilot-v1 번호는 이전과 같고 `video-grounding-v2`·`video-grounding-v3`는 각각 1부터 시작합니다. 저장된 행은 고치지 않습니다.
  - 승인 추적: 새 승인은 `approve_content(runId, approved_by)` 호출자가 넘긴 승인자 `approvedBy`와 승인 시각 `approvedAt`을 함께 기록합니다. 두 필드는 함께 있거나 함께 없어야 하고, 추적 필드가 있으면 그 행은 `approved` 상태의 Grounding 행이어야 합니다. 승인은 Human Evaluation 결과와 무관합니다.
  - `approvedBy`: 앞뒤 공백을 제거한 1~100자이며 제어 문자, 줄·문단 구분 문자, 서식 문자, 짝 없는 surrogate는 허용하지 않습니다. 호출자가 제공한 식별 문자열일 뿐 사용자 인증이나 GitHub·OS·계정 identity 확인이 아닙니다. 금지 문자 판정은 실행 중인 Python의 Unicode 데이터베이스를 따르므로, 이후 Unicode 버전에서 새로 분류된 문자는 Python 버전에 따라 판정이 달라질 수 있습니다.
  - `approvedAt`: 이 저장소가 생성하는 `YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00` 형식(UTC, 소수 초는 없거나 정확히 6자리)만 유효합니다. `Z`, 다른 offset, 공백 구분자, 초 생략 등은 거부합니다. JSON Schema는 이 문자열 pattern만 제한하며, 존재하지 않는 날짜·시각은 Python runtime 검증(`datetime.fromisoformat`)이 거부합니다. `approvedBy`의 Unicode category와 strip 전 원본 검사 규칙도 schema가 아니라 runtime 검증에서만 확인합니다.
  - 기존 승인: 승인 추적 도입 전에 기록된 `approved` 행(첫 Pilot 포함)에는 두 필드가 없으며 계속 유효합니다. 이 승인들의 승인자와 승인 시각은 사후에 확인할 수 없고, migration·backfill이나 재승인으로도 채우지 않습니다.
  - 재승인: 이미 `approved`인 행을 다시 승인하면 다른 `approved_by`를 넘겨도 기존 값을 바꾸지 않고 파일도 다시 쓰지 않으며 저장된 행을 그대로 반환합니다. 호출자는 반환된 행의 `approvedBy`(legacy 행은 필드 없음)로 결과를 확인할 수 있습니다.
  - 거부: 추적 필드가 한쪽만 있거나, 형식이 잘못됐거나, `approved`가 아닌 행에 있으면 승인, standalone Quiz와 offline 집계(`aggregate_pipeline`) 모두에서 거부합니다.
  - 한계: 이 필드는 누가 승인했다고 기록되었는지를 남길 뿐이며, 결과 파일의 변조 방지·서명을 제공하지 않습니다.
  - Pilot v2 (`configs/pilot-v2.yaml`, Quiz `pilot-v2`, Grounding `video-grounding-v3`; 중단된 첫 실행은 `video-grounding-v2`): pilot-v1과 별개의 실험이며 기존 pilot-v1 결과·manifest·설정은 바꾸지 않습니다. 고정 입력 manifest는 pilot-v1에서 영상·Quiz prompt 버전별, pilot-v2에서 영상·Quiz prompt 버전·repetition별이라 v1 항목과 공존합니다. pilot-v1 manifest의 key·기록 형식은 바꾸지 않았습니다. `video-grounding-v2`·`video-grounding-v3` Grounding은 `approve_content`로 승인할 수 없고 `review_grounding(runId, reviewed_by, checklist)`로만 사람 검수를 기록합니다. 체크리스트의 의미는 [평가 기준](rubric.md)을 따릅니다.
  - `groundingReview`: `factualAccuracy`, `keyInformationCoverage`, `factsConsistency`, `koreanConsistency`, `contentTextContractCompliance`(각각 사람이 입력한 `pass|fail|uncertain`, `null` 불가), 선택 `reviewNote`(문자열 또는 `null`), `reviewedBy`(`approvedBy`와 같은 규칙), `reviewedAt`(`approvedAt`과 같은 형식). 성공한 `video-grounding-v2`·`video-grounding-v3` 행에만 있고, 원본 evaluation의 `contentText` SHA-256이 행의 값과 같을 때만 기록하며 이미 검수된 행은 덮어쓰지 않습니다.
  - 유도 상태: 다섯 항목이 모두 `pass`면 `contentTextApprovalStatus=approved`이고 `approvedBy`/`approvedAt`에 검수자·검수 시각을 같은 값으로 기록합니다. 하나라도 `fail`·`uncertain`이면 `rejected`이며 승인 추적 필드는 없습니다. `rejected`는 품질 판정이지 technical failure가 아닙니다. rejected Grounding으로는 Quiz를 실행하지 않습니다(Provider 호출·Quiz 행 없음).
  - 조건당 성공 후보 하나: 검수가 필요한 Grounding 조건(`videoId`·`method`·`model`·`repetition`·Grounding `promptVersion`, 즉 `video-grounding-v2` 또는 `video-grounding-v3`)에 성공한 Grounding이 하나라도 있으면 검수 상태(미검수·approved·rejected)와 관계없이 그 조건의 새 Grounding 생성을 Provider 호출 전에 거부하며 아무것도 기록하지 않습니다. 다음 attempt는 technical failure 뒤에만 허용합니다. 이미 같은 조건에 성공 행이 둘 이상 있으면 그중 하나를 자동으로 고르지 않고 storage integrity 오류로 모든 실행·승인·검수를 중단합니다(`review_grounding`도 같은 조건의 다른 성공 후보가 있으면 거부). 재생성이나 후보 선택은 별도의 사람 결정 사항입니다. pilot-v1·legacy Grounding의 재시도·attempt 규칙은 바꾸지 않습니다.
  - Quiz source 버전: Quiz의 `promptVersion`이 원천 Grounding 버전 그룹을 정합니다. `pilot-v1`은 legacy `null`·`video-grounding-v1`, `pilot-v2`는 `video-grounding-v3` 원천만 허용하고, 그 밖의 조합이나 알 수 없는 버전은 허용하지 않습니다. timestamp 문제로 중단된 첫 Pilot v2 실행의 `video-grounding-v2` Grounding은 더 이상 Pilot v2 Quiz 원천이 아닙니다. standalone Quiz는 다르면 Provider 호출과 고정 입력 manifest 기록 전에 거부하고, offline 집계(`aggregate_pipeline`·`compare_pair`)도 같은 규칙으로 거부합니다. offline 집계는 선택된 v2 Grounding 조건의 성공 행이 정확히 하나일 때만 집계합니다.
  - repetition별 고정 입력: Pilot v2의 각 repetition은 자기 Grounding·사람 검수·승인 `contentText`에서 시작하므로 repetition마다 다른 승인 `contentText`를 고정할 수 있고, 같은 repetition의 Quiz A/B는 같은 `contentText`만 사용합니다. Pilot v2 Quiz는 같은 repetition의 승인 Grounding만 원천으로 사용하며, 다르면 Provider 호출·Quiz 행·고정 입력 manifest 기록 전에 거부합니다. offline 집계·Judge·Human 평가의 Grounding/Quiz repetition 일치 검사는 그대로입니다. pilot-v1은 기존처럼 영상당 하나의 고정 입력을 repetition 간에 공유합니다.
  - 결과 디렉터리 경계: 한 Pilot v2 실험의 Grounding·Quiz 결과는 하나의 `--results-dir`에서 관리합니다. 조건당 성공 후보 규칙, rejected 재생성 금지, attempt 번호와 storage integrity 검사는 결과 디렉터리 단위로 적용되며 다른 디렉터리의 결과는 보지 않습니다.
  - 무결성: 실행·승인·검수 전 storage integrity 검사, standalone Quiz와 offline 집계는 `groundingReview`와 상태의 일치를 확인합니다. 알 수 없거나 형식이 잘못된 Grounding `promptVersion`(legacy `null`, `video-grounding-v1`, `video-grounding-v2`, `video-grounding-v3`만 허용), 같은 v2·v3 조건의 성공 행 여러 개, 검수 없는 v2·v3 `approved`, `fail`·`uncertain`이 있는 `approved`, 검수 없는 `rejected`, 모두 `pass`인 `rejected`, v2·v3가 아닌 행의 `groundingReview`는 거부합니다. pilot-v1·legacy 행은 기존 규칙 그대로입니다.
- `results/quiz-generation.jsonl`: `(videoId, method, model, promptVersion, contentTextSha256, repetition, attempt)` 실행마다 한 행. `benchmarkType=quiz_generation`. 영상별(Pilot v2는 영상·repetition별) **동일한 고정 `contentText`**의 SHA-256을 두 모델에 공통으로 사용합니다. `sourceGroundingRunId`는 원천 Grounding 결과를 추적할 때만 사용하며, 입력 내용은 제한된 로컬 파일로 보관합니다. API·Parser·Benchmark 내부 Quiz 형식/계약 검사 상태와 질문별 정답 정확성·유일성·근거의 정답 지지·한국어 품질을 구분해 기록합니다.
- `results/end-to-end.jsonl`: Direct는 `(videoId, method, model, promptVersion, repetition, attempt)` 실행마다 한 행입니다. `benchmarkType=end_to_end`. 두 단계 방식의 과거 결과는 `groundingRunId`와 `quizRunId`를 참조합니다. 현재 두 단계 자동 실행은 사람 승인 전 Quiz 호출을 막기 위해 거부하며, Grounding과 승인 후 Quiz를 별도로 실행합니다. 직접 Quiz는 두 ID가 모두 `null`이며 질문별 검수 결과를 이 행에 남깁니다. 직접 Quiz의 `beCompatibility`는 항상 `not_applicable`입니다. 직접 Quiz의 `validatorStatus`는 parsing 성공 후 공통 Quiz 구조 계약(문제·보기 수, 정답 인덱스, 빈 문자열·중복, promptVersion 등)에 실패하면 `fail`(`errorCategory=quiz_contract_error`), 구조 계약을 통과하면 `not_run`입니다. `not_run`은 고정 `contentText`가 없어 evidence/contentText 검사가 적용되지 않는다는 뜻이며 통과를 뜻하지 않습니다. `parseStatus=fail`이면 `validatorStatus=not_run`, `errorCategory=parse_error`입니다. 세 경우 모두 Provider 응답을 정상 수신했다면 `apiStatus=success`를 유지하며 출력 결과로 보존합니다.

`src.offline_aggregator.aggregate_pipeline(results_dir, grounding_run_id, quiz_run_id)`는 기존 Grounding·standalone Quiz 결과와 승인된 원본 contentText의 SHA-256을 읽어 연결한 **파생 결과를 반환**합니다. 새 API 호출이나 자동 두 단계 실행을 하지 않으며 결과 파일도 저장하지 않습니다. `compare_pair(results_dir, quiz_a_run_id, quiz_b_run_id)`는 같은 영상·반복·Grounding run·승인 SHA-256·Quiz prompt 버전인지 추가로 확인합니다. 두 단계의 API 성공과 Human Evaluation은 별도이며 새로운 종합 품질 판정을 만들지 않습니다. `apiLatencySumMs`는 측정된 API latency 두 값의 합일 뿐 Human Review·Approval 시간을 포함하지 않습니다. Provider별 token usage는 합산하지 않고 단계별로 보존합니다. 두 단계의 비용이 있을 때 계산한 `selectedPipelineEstimatedCostUsd`는 선택된 run의 **추정 비용**이며 실제 청구액이나 실패 attempt까지 포함한 총지출이 아닙니다. A/B는 하나의 Grounding을 공유할 수 있으므로 실험 전체 비용에서 이를 중복 집계하지 않아야 합니다.

공통 식별·호출 필드는 `runId`, `videoId`, `method`, `model`, `promptVersion`, `repetition`, `attempt`, `startedAt`, `apiStatus` (`success|error|not_run`), `errorCategory`, `httpStatus`, `providerErrorCode`, `retryStopReason`, `latencyMs`, `inputTokens`, `outputTokens`, `thinkingTokens`, `estimatedCostUsd`, `pricingReference`입니다. `repetition`은 독립적인 실험 반복, `attempt`는 동일한 실험 조건의 기술적 실행 시도 번호입니다. Grounding은 `videoId`·`method`·`model`·`repetition`·Grounding `promptVersion`(`null`은 `video-grounding-v1`과 같은 그룹), Quiz Generation은 `videoId`·`method`·`model`·`promptVersion`·`contentTextSha256`·`repetition`이 모두 같을 때 다음 번호를 부여합니다. Direct End-to-End는 `videoId`·`method`·`model`·`promptVersion`·`repetition`을 사용하며 과거 two-stage End-to-End는 기존 키를 유지합니다. 기존 행에 `attempt`가 없으면 1로 해석하고 파일은 변경하지 않습니다. `httpStatus`는 HTTP 오류에서 확인된 숫자만 기록합니다. 선택적 `providerErrorCode`는 Gemini Interactions API의 JSON 오류 응답에서 안전한 허용 목록의 `error.code`를 추출한 경우에만 기록하며, 없거나 확인할 수 없으면 `null`입니다. `retryStopReason`은 429/5xx 응답 뒤 다음 retry가 실행 제한(live guard)으로 차단된 경우에만 `live_guard`이고 그 외에는 `null`입니다. 이때 `errorCategory`·`httpStatus`·`providerErrorCode`는 마지막 실제 Provider 실패를 유지하며, 첫 HTTP 호출 전 거부는 기존처럼 `errorCategory=live_guard`, `retryStopReason=null`입니다. 따라서 `null`은 guard가 관여하지 않았다는 뜻이 아니며 성공, retry 소진, retry 대상이 아닌 오류(timeout·network 포함), fixture, local transcript, 첫 HTTP 호출 전 guard 거부를 모두 포함합니다. 실행 제한이 HTTP 요청을 막아 끝난 run을 집계하려면 `errorCategory=live_guard` 또는 `retryStopReason=live_guard`인 행을 함께 봐야 합니다. 기존 행에 필드가 없어도 유효하지만, `retryStopReason`이 없는 기존 행은 `null`과 달리 기록되지 않은 것이므로 과거 retry가 guard로 중단됐는지는 복원할 수 없습니다. 오류 응답 본문·자유형 오류 message·요청/응답 헤더·API Key는 저장하지 않습니다. transcript Grounding처럼 적용되지 않는 값과 Provider가 usage를 반환하지 않은 값은 `null`로 두며 `0`으로 채우지 않습니다. 로컬 원본의 경로·전체 텍스트·API Key는 공유 결과 행에 넣지 않습니다.

실제 Provider가 입력·출력 usage를 반환하고 운영자가 모델별 단가와 출처를 제공한 경우 `estimatedCostUsd`는 usage × 입력/출력 단가의 추정치로 기록합니다. Gemini는 별도 thought token을 출력에 더하고, OpenAI는 reasoning token이 포함된 `outputTokens`를 그대로 사용합니다. usage가 없으면 비용은 `null`입니다. 이 값은 실제 청구액이 아니며 영상 입력 토큰의 사전 추정이나 캐시·계정별 요금 차이를 보장하지 않습니다.

사람 판정 값은 `pass|fail|uncertain`입니다. 평가하지 않았거나 승인된 ground truth가 없어 확정할 수 없는 값은 `null`입니다. `uncertain`과 `null`을 성공으로 합산하지 않습니다. 새 실행·approval 전 storage integrity 검사와 offline 집계(선택된 Grounding·Quiz 행)는 저장된 사람 판정 값을 [`run-result.schema.json`](run-result.schema.json) 계약으로 검사하고 위반 시 중단합니다. 판정 값은 정확히 `pass|fail|uncertain|null`(대소문자·공백 차이, bool·숫자 거부), `reviewNote`는 문자열 또는 `null`, `questionReviews` 항목의 `questionIndex`는 0 이상의 정수여야 합니다. `groundingFacts`·`questionReviews`는 있으면 배열이어야 하며(`null` 거부), 항목에 schema에 없는 key가 있으면 거부합니다. 필드 누락·빈 배열·`null` 판정은 허용하므로 평가 완료 여부를 강제하지 않습니다. `contentTextApprovalStatus=approved`는 별도의 사람 승인으로, 사실별 판정이나 API 성공에서 자동 유도하지 않습니다(`video-grounding-v2`는 사람이 입력한 `groundingReview` 다섯 항목에서만 `approved`/`rejected`를 유도). Standalone Quiz는 원천 Grounding `runId`, 성공 상태, 영상 ID, 승인 상태 및 승인 해시를 확인하고, 원본 evaluation의 텍스트와 실제 입력 텍스트의 SHA-256이 모두 승인 해시와 같을 때만 실행합니다. 고정 입력 manifest도 계속 검사합니다. 새 Grounding에서 즉시 Quiz로 이어지는 두 단계 E2E 실행은 차단하며 Grounding과 승인 후 Quiz를 별도로 실행합니다. Direct Quiz는 텍스트 승인의 대상이 아닙니다. `evidenceTextContained`는 문자열 검사이고 `evidenceSupportsAnswer` 및 `videoGrounding`은 별도의 사람 판정입니다. timestamp의 값이 있다는 것과 `timestampAccurate=pass`는 다른 의미입니다.

기존 필드명 `beCompatibility`는 실제 Cking-BE validator의 실행 결과가 아닙니다. Standalone Quiz의 `beCompatibility=pass`는 Benchmark 내부 로컬 validator 통과만 뜻합니다. 로컬 검사는 JSON 파싱, `questions` 배열, 필수 문자열 필드(`question`, `explanation`, `sourceEvidence`), `options` 배열의 문자열, 정수 `correctOptionIndex`, promptVersion 일치, 설정된 문제·보기 수, 0-based 정답 인덱스 범위, 빈 문자열·중복 질문/보기, 입력 `contentText` 길이 제한, `sourceEvidence`의 고정 `contentText` 내 정확한 문자열 포함 여부를 확인합니다. 실제 정답 정확성·근거의 의미적 지지, Cking-BE production DTO/domain validation 및 서비스 integration compatibility는 검증하지 않습니다. 기존 Quiz A/B의 `parseStatus=pass`, `validatorStatus=pass`, `beCompatibility=pass`도 로컬 검사 통과로만 해석하며 결과를 재판정하지 않습니다. `beCompatibilityRate`는 이 **Benchmark-local validation pass rate**이며 실제 BE 호환 성공률이 아닙니다. Grounding에는 해당 필드가 없습니다. Direct Quiz는 고정 `contentText`가 없어 동일한 evidence/contentText 검사를 적용하지 않으므로 `beCompatibility=not_applicable`로 기록합니다. 이는 Direct Quiz의 품질·정답 정확성·실제 BE 호환성 통과를 뜻하지 않습니다.

Grounding 행의 `omission`은 승인된 ground truth의 핵심 의미가 `contentText` 또는 `groundingFacts`에 충분히 반영됐는지에 대한 영상별 사람 판정입니다. 모두 반영되면 `pass`, 하나라도 양쪽 모두에서 누락되면 `fail`입니다. `hallucination`은 두 출력의 실질적인 주장을 원본 영상과 대조한 사람 판정입니다. 모두 영상에서 확인되면 `pass`, 영상에서 확인되지 않는 주장이 하나라도 있으면 `fail`입니다. Ground truth에 없다는 이유만으로 환각 처리하지 않습니다. 두 필드 모두 검토 후 신뢰성 있게 판단하기 어려우면 `uncertain`, 미검토라면 `null`이며 자동으로 채우지 않습니다. 세부 기준은 [평가 기준](rubric.md)을 따릅니다.

## Pilot v2 Human Quiz 평가 결과

Pilot v2 Quiz A/B/C의 블라인드 Human Evaluation은 `results/human/<sessionId>/`에만 저장하며 Pilot summary 행과 [`run-result.schema.json`](run-result.schema.json)은 바꾸지 않습니다. 이 하위 디렉터리는 Pilot 저장 무결성 검사 대상이 아닙니다. 세션 생성과 import는 같은 results 디렉터리의 결과 변경 작업과 같은 `.pilot.lock` 잠금을 사용합니다.

- `session.json`(비공개): seed, 순서 방식(`sha256-rank-v1`), blindId·videoRef와 원본 run의 대응(`benchmarkType`, `runId`, 조건 A/B/C, `videoId`, `repetition`, `promptVersion`, `evaluationSha256`, A/B는 `sourceGroundingRunId`·`contentTextSha256`), `blindExportSha256`, 평가 제외 기록(`notEligible`: 조건, 사유(`notRun`은 기록된 시도 없음, `noApiSuccess`는 시도는 있으나 API 성공 없음, `parseFailed`, `questionCountMismatch`, `invalidQuestionStructure`), 시도별 `runId`·`attempt`·`apiStatus`·`errorCategory`·`httpStatus`·`parseStatus`·`validatorStatus`·`questionCount`). 세션 생성 시 마지막에 쓰므로 이 파일이 없는 디렉터리는 불완전한 세션으로 거부합니다.
- `blind-export.json`(평가자에게 전달): `format`, `sessionId`, `rubricVersion`, 영상별 `videoRef`·`youtubeUrl`과 세트별 `blindId`, 문항(`questionIndex`, `question`, `options`, `correctOptionIndex`, `explanation`, `sourceEvidence`), 비어 있는 `setReview`·`questionReviews`. 허용 목록에 있는 필드만 포함하며 조건·model·runId·contentText·promptVersion·repetition·attempt·사용량·validator 정보·파일 경로는 넣지 않습니다.
- `evaluation.json`(import 결과): `reviewedBy`, `reviewedAt`(import 시각, UTC), 세트별 원본 run 대응과 `setReview`·`questionReviews`. 한 번만 만들며 덮어쓰지 않습니다. 다시 평가하려면 새 세션을 만듭니다.

순서는 seed로 재현합니다. 영상은 `sha256(seed|video|videoId)`, 영상 안의 세트는 `sha256(seed|set|runId)` 순으로 정렬하고 그 순서대로 `V01…`, `S001…`을 붙이며, 실제 순서와 대응은 `session.json`에 저장합니다. Human 세션은 원칙적으로 Pilot v2 생성 절차를 마친 뒤 만들며, 세션 생성은 모든 조건의 실행 완료를 요구하지 않습니다. 세션 생성 뒤 Pilot v2 결과가 바뀌어 평가 대상 배정이 달라지면 import를 거부하므로 새 세션을 만듭니다. import는 세션·blind export SHA, videoRef와 영상별 blindId 구성의 정확한 일치(누락·추가·중복·이동 거부), 문항·YouTube URL 일치, `questionIndex` 0·1·2, 판정 허용값과 null 금지, 허용되지 않은 필드, `reviewedBy`, 원본 Pilot 행의 method·model·videoId·repetition·promptVersion·`questionCount=3`(A/B는 원천 Grounding과 contentText SHA 포함)과 `evaluation/<runId>.json` SHA, 각 blindId에 표시된 문항과 대응된 원본 run 문항의 일치, 저장된 seed와 현재 Pilot v2 결과로 다시 계산한 (blindId, videoRef, 원본 run) 배정이 `session.json`과 같은지(같은 문항을 가진 run끼리 배정이 바뀐 경우도 거부)를 모두 확인한 뒤에만 `evaluation.json`을 만듭니다.

Pilot v2의 Human 판정 기준 자료는 `evaluation.json`입니다. Pilot v2 summary의 `questionReviews`는 실행 시 만든 빈 값 그대로 두며 Human 판정을 쓰지 않습니다. Pilot v1은 기존처럼 summary의 `questionReviews`가 기준입니다. Judge와 offline 집계는 아직 이 결과를 읽지 않습니다.

## Production 자동 Grounding machine validation (Issue #37)

Production 자동 Grounding은 사람 검수를 거치지 않으므로 Pilot의 Human 승인을 쓰지 않고, 별도의 versioned machine validation으로 production 사용 가능 여부를 판정합니다(`src/production_grounding.py`). Provider를 호출하지 않으며 저장된 결과를 읽기만 합니다.

| 구분 | 의미 |
| --- | --- |
| Human 승인 (`contentTextApprovalStatus=approved`, `approvedBy`, `approvedAt`, `groundingReview`) | 사람이 원본 영상과 대조한 Pilot Grounding 검수를 통과했다는 Pilot 계약. 그대로 유지하며 machine validation이 읽거나 쓰지 않습니다. |
| machine validation `pass` | 저장된 Grounding 결과가 해당 validation version의 구조·원천 식별·`contentText` hash·저장 무결성 계약을 통과했다는 뜻. production 사용 가능 판정의 전제입니다. |

**Machine validation은 원본 영상 기준의 사실 정확성을 검증하지 않습니다.** 사실 정확성, 핵심 정보 포함, timestamp가 가리키는 실제 위치, `evidenceType`의 실제 일치를 확인하지 않으며, 통과한 `contentText`도 verified ground truth가 아닙니다. 구조적으로 올바르지만 영상과 다른 내용은 통과할 수 있습니다.

- 대상 artifact: Grounding summary 행, `evaluation/<runId>.json`(`contentText`, `facts`), `raw/<runId>.json`(Provider 완료 metadata). validation version `production-grounding-validation-v1`은 `gemini_video` / `gemini-3.8-flash` / `video-grounding-v3` Grounding만 다룹니다(`model`이 없거나 다른 모델이면 거부).
- 기록: Grounding 행에 필드를 추가하지 않고 별도 객체 `{status: pass|fail|not_run, version, groundingRunId, contentTextSha256, validatedAt, failureReasons}`로 만듭니다. `fail`에만 사유가 있고, `validatedAt`은 실제로 존재하는 canonical UTC 시각이어야 합니다. Quiz 생성에 실제로 사용한 기록은 아래 Production Quiz evidence snapshot에 저장합니다. 기존 Pilot 결과는 migration하지 않습니다.
- production 사용 가능 조건(`require_production_grounding_eligible`, 모두 만족해야 하며 아니면 사유와 함께 거부): 지원하는 validation version의 `pass` 기록이 같은 `runId`와 실제 `contentText` SHA-256을 가리킴, 요청한 `videoId`와 일치, `apiStatus=success`·`errorCategory=null`, raw metadata가 Pilot 저장과 같은 계약(`validate_raw_metadata`)을 만족하는 Gemini 완료 기록(fixture `not_run`과 Provider 미완료 거부), 비어 있지 않고 UTF-8로 표현할 수 있는 `contentText`, `facts` 필드가 배열로 존재(누락을 빈 배열로 보지 않음. `video-grounding-v3` 응답에 `facts`가 없으면 evaluation 파일에도 `facts` key를 쓰지 않음), 각 fact가 저장 계약(5개 key, 비어 있지 않은 문자열, `evidenceType` 허용값, timestamp는 `null` 또는 0 이상·순서·알려진 영상 길이 이내)을 만족, 행의 `groundingFacts`와 일치, 행의 `contentTextSha256`과 일치. 저장된 `pass`만 믿지 않고 artifact를 다시 검사합니다. Human 승인은 이 기록을 대신하지 않습니다.
- 정하지 않은 정책: facts 최소 개수(빈 배열도 구조상 허용), `null` timestamp 금지 여부, `contentText` 최소 길이, 영상 길이 metadata가 없을 때의 정책(현재 계약처럼 상한 검사를 하지 않음), 의미·사실 검증, retry·재생성, Creator reject 이후 동작.

## Production Quiz 생성과 output gate (Issue #39)

machine eligibility를 통과한 Grounding의 정확한 `contentText`로 Gemini Quiz를 만들고, GPT Pointwise Judge로 보낼 수 있는지(Judge-ready) 판정합니다(`src/production_quiz.py`, `configs/production.yaml`). Pilot과 별도 실행 경로이며 Pilot Human 승인, repetition, fixed-content manifest를 쓰지 않고 Pilot 결과에 쓰지 않습니다. GPT Judge 호출, Creator 승인, 게시, 재생성은 포함하지 않습니다.

| 구분 | 의미 | 의미하지 않는 것 |
| --- | --- | --- |
| Grounding machine eligibility | Grounding 구조·식별·hash·저장 무결성 계약 통과 | 영상 사실 승인 |
| Quiz output gate 통과(Judge-ready) | Quiz 구조와 `sourceEvidence`의 정확한 포함, provenance 무결성 통과 | 정답·설명의 의미적 정확성, 한국어·오답 보기 품질, 의미상 중복, 학습 가치(후속 GPT Judge 영역) |
| Judge-ready | GPT Pointwise Judge 입력으로 보낼 수 있음 | Creator 최종 승인 |

- 생성 조건: `gemini-3.8-flash`, `thinking_level: medium`, `max_output_tokens: 8192`(Pilot v2 Quiz A와 같은 값), `fixed_content_text`, prompt `production-quiz-v1`(Pilot prompt와 별도 registry이며 Pilot config는 선택할 수 없음), 출력 계약 `production-quiz-output-v1`(정확히 3문항, 각 4보기, 0-based `correctOptionIndex`, 문항 필드는 `question`·`options`·`correctOptionIndex`·`explanation`·`sourceEvidence`만 허용). Provider는 live(`is_actual_api`)여야 하며 fixture 출력은 production 결과가 아닙니다. 요청 내용을 바꾸는 생성 설정(`thinking_level`, `max_output_tokens`)은 Provider가 실제로 보낼 값(`request_settings`)이 config와 정확히 같아야 하며, 다르거나 확인할 수 없으면 Provider 호출과 저장 전에 거부합니다. retry 횟수·비용 상한·timeout 같은 실행 guard는 fingerprint에 넣지 않습니다.
- 입력: `require_production_grounding_eligible`가 반환한 `contentText`를 그대로 보냅니다. 미적격이면 Provider를 호출하지 않고 아무것도 저장하지 않습니다.
- 저장(`results/production/`, 모든 파일은 한 번만 만들고 바꾸지 않음):
  - `sources/<sha256>.json`: evidence snapshot. Grounding 행, evaluation(`contentText`, `facts`), raw metadata, 사용한 machine validation record, `videoId`, `sourceGroundingRunId`, `contentTextSha256`, 영상 길이. 파일 이름이 내용의 SHA-256이며, 원본 Grounding이 바뀌거나 지워져도 이 snapshot으로 근거를 복원합니다.
  - `operations/<operationId>/operation.json`: 생성 작업 ID, 입력·설정 fingerprint(`videoId`, `sourceGroundingRunId`, `contentTextSha256`, Grounding validation version, model, method, prompt version, 출력 계약 version, 생성 설정(`thinking_level`, `max_output_tokens`)의 canonical JSON SHA-256), snapshot SHA-256.
  - `operations/<operationId>/attempts/<n>/started.json`(요청 전), `raw.json`(canonical raw metadata), `output.json`(모델 출력 원문, 수정하지 않음), `result.json`(마지막에 기록: API·parse·validator·output gate 상태와 사유, `errorCategory`, latency, token, 추정 비용, raw·output SHA-256).
- 작업 ID와 재시도: 새 Quiz 생성(재생성 포함)은 새 operation ID입니다. 같은 operation을 다른 입력·설정·evidence snapshot으로 다시 쓰는 것은 거부합니다. Provider 성공(`apiStatus=success`)이 기록된 operation은 output gate 결과와 관계없이 다시 실행하지 않습니다. 같은 operation의 다음 attempt는 이전 attempt가 모두 검증된 기술적 실패일 때만 명시적으로 실행할 수 있습니다. 검증된 기술적 실패란 `started.json`과 `result.json`이 모두 있고, 두 기록의 식별 필드가 `operation.json`에서 다시 도출한 값과 일치하며, `apiStatus`가 정확히 `error`이고 `errorCategory`가 있으며 raw·output 파일이 없는 attempt입니다. `apiStatus`가 없거나 `not_run`·알 수 없는 값, 형식이 깨진 기록, `started.json` 없는 `result.json`, 식별 불일치는 모두 Provider 호출 전에 거부합니다. 자동 재시도·재생성은 없습니다. 서비스 retry 횟수와 비용 상한은 정하지 않았고, live 요청은 기존처럼 호출 수·비용 상한·timeout·HTTP retry 횟수를 실행할 때 명시해야 합니다.
- 불확실·손상 상태: `started.json`만 있고 `result.json`이 없는 attempt는 요청이 Provider에 도달했을 수 있으므로 `uncertain`이며, 완료나 Judge-ready로 보지 않고 자동으로 다시 실행하지 않습니다. `result.json`만 있거나 기록 사이 식별이 맞지 않으면 `corrupted`입니다. 두 파일이 모두 없는 attempt 디렉터리는 요청 전이므로 attempt로 보지 않습니다. 처리 방법은 사람이 정합니다.
- output gate: Provider 완료, parse, 허용 필드, promptVersion 일치, 문항 수, 문항 구조(Pilot과 같은 규칙: 비어 있지 않고 중복 없는 질문·보기, 보기 수, 정답 index 범위, 비어 있지 않은 설명·근거), `sourceEvidence`가 `contentText`의 정확한 부분 문자열. `sourceEvidence`를 따옴표 제거·trim·수정하지 않으며 `evidence_not_in_content`는 출력과 실패를 보존한 채 Judge-ready가 되지 않습니다. 다른 구조 실패(`quiz_contract_error`, `parse_error`)도 같습니다.
- Judge-ready 판정(`require_judge_ready`): 저장된 상태를 믿지 않고 operation fingerprint, 모든 attempt의 `operation.json` → `started.json` → raw·output → `result.json` 연결(각 기록의 operation·attempt·fingerprint·snapshot·Grounding run·`contentTextSha256`·model·prompt version·시작 시각 일치, 허용 필드, 파일 SHA-256, raw metadata 계약), snapshot SHA-256과 그 evidence의 Grounding eligibility, snapshot의 `contentText` 기준 output gate를 다시 확인합니다. 통과하면 `contentText`, 문항(`questionIndex` 포함)과 provenance(operation·fingerprint·Grounding run·`contentTextSha256`·validation version·snapshot·output SHA-256)를 돌려주며 생성 모델 정보는 넣지 않습니다.
- 정하지 않은 정책: 자동 재생성 횟수, Creator reject 후 재생성, 서비스 retry 횟수, production 비용 상한, `contentText` 최대 크기, Judge 실패·uncertain 처리, 게시 정책, 영상 사실 검증.

## Production GPT Pointwise Judge (Issue #41)

Judge-ready production Quiz를 GPT Pointwise Judge로 평가하고 문항별 결과를 저장합니다(`src/production_judge.py`, `configs/production-judge.yaml`). Pilot Judge의 OpenAI client(인증, usage·비용 계산, cost guard, HTTP retry), Pointwise prompt·schema, `validate_pointwise`(strict JSON, 중복 key 거부)를 그대로 재사용합니다. Pilot Judge([Judge 프로토콜](judge-protocol.md))의 Pairwise, A/B, Human 비교는 쓰지 않으며 변경하지 않습니다.

- **평가 범위**: Judge는 Quiz를 정확한 `contentText` 기준으로만 평가합니다. 영상을 보지 않으므로 Grounding과 원본 영상의 사실 일치, Grounding의 핵심 정보 누락이나 사실 오류, timestamp·`evidenceType`의 정확성을 검증하지 않습니다. 잘못된 Grounding에 충실한 Quiz도 `pass`일 수 있습니다.
- **설정**: OpenAI `gpt-6.1-sol`, reasoning `medium`, `max_output_tokens` 8192, prompt version `production-pointwise-judge-v1`, 기존 rubric·schema(`judge-rubric-v1`, `judge-schema-v1`)의 7개 항목(`textAnswerCorrect`, `uniqueAnswer`, `evidenceSupportsAnswer`, `questionClarity`, `koreanQuality`, `distractorQuality`, `textFaithfulness`)마다 `pass | fail | uncertain`과 reason. API key는 `OPENAI_API_KEY` 환경변수에서 읽으며 저장하지 않습니다.
- **입력**: `ProductionQuizRunner.require_judge_ready()`를 통과한 Quiz만 평가합니다. 통과하지 못하면 HTTP 요청과 저장이 없습니다. prompt에는 `contentText`와 문항(`questionIndex`, `question`, `options`, `correctOptionIndex`, `explanation`, `sourceEvidence`)만 들어가며, operation ID, hash, 생성 모델, validation 상태 같은 provenance는 저장소에만 남습니다.
- **실행 guard**: timeout, 호출당·평가 전체 비용 상한, 입력 token 추정치, `Retry-After` 대기 상한(`maxRetryAfterSeconds`, 0 이상의 유한한 수), 단가(출처·확인일 포함)는 기본값이 없으며 실행마다 명시합니다. 누락·잘못된 값, API key 없음은 저장과 요청 전에 거부합니다. 첫 실제 호출 전에 OpenAI 공식 가격과 token·reasoning 계산을 다시 확인합니다.
- **재시도**: technical attempt 하나 안에서 429/5xx에 유효한 `Retry-After`가 있고 그 값이 `maxRetryAfterSeconds` 이하일 때만 그만큼 기다린 뒤 HTTP 재시도 1회. 상한을 넘으면 기다리거나 재시도하지 않고 종료합니다. 평가당 technical attempt 최대 2, HTTP 요청 최대 4이며 다음 attempt는 자동으로 시작하지 않습니다.

| attempt 결과 | 해당 경우 | 다음 동작 |
| --- | --- | --- |
| `completed` | 유효한 응답(`fail`·`uncertain` 판정 포함) | 평가 완료. 같은 평가를 다시 호출하지 않음 |
| `retryable_failure` | malformed JSON, schema·문항 index·빈 reason 오류, `incomplete`, 해석할 수 없는 응답, HTTP 재시도 후에도 429/5xx이고 마지막 응답에 상한 이내의 유효한 `Retry-After`가 있음 | 원문·오류 보존. 상한 안에서 명시적 다음 attempt만 가능 |
| `terminal_failure` | refusal, 일반 4xx, 마지막 응답에 유효한 `Retry-After`가 없거나 `maxRetryAfterSeconds`를 넘는 429/5xx(HTTP 재시도 후 포함), 실행 guard | 다시 실행하지 않음 |
| `uncertain` | timeout, network 오류(요청이 처리됐을 수 있음) | 자동 재실행 없음. 사람이 처리 |

시작 기록이나 요청 예약만 있고 결과 기록이 없는 attempt도 `uncertain`입니다. 유효한 판정과 함께 usage가 잘못된 응답은 완료로 저장하되 비용은 알 수 없음(`null`)으로 둡니다. attempt 결과는 저장된 오류 분류와 HTTP 요청 기록(상태, `Retry-After`, 대기)에서 다시 계산하며, 저장된 결과와 다르면 기록이 손상된 것으로 봅니다.

- **저장**(`results/production/operations/<quizOperationId>/judge/<evaluationId>/`, 모든 파일은 한 번만 만들고 바꾸지 않음): `evaluation.json`(identity: Quiz operation·attempt·fingerprint·output·result SHA-256, Grounding run, `contentTextSha256`, snapshot SHA-256, validation version, Quiz prompt·출력 계약 version과 Judge provider·model·reasoning·`max_output_tokens`·prompt·rubric·schema version, Pointwise 계약 hash, 요청 본문 SHA-256; identity의 fingerprint; 고정 retry 정책과 실행의 timeout·비용 상한·입력 추정치·단가), `request.json`(정확한 요청 본문), `attempts/<n>/started.json`, `attempts/<n>/requests/<k>.json`(HTTP 요청마다 전송 전에 기록하는 예약), `attempts/<n>/output.json`(모델 출력 원문), `attempts/<n>/result.json`(마지막에 기록: 결과, 오류 분류, HTTP 요청 기록과 예약·정산 비용, usage, 출력 SHA-256, 문항별 판정).
- **재개와 무결성**: 같은 평가는 Quiz artifact, Judge 설정, 요청 본문, 실행 정책이 모두 같을 때만 이어갑니다. 저장된 실행 정책은 읽을 때도 새 실행과 같은 규칙으로 모든 값을 검증하고, 그 SHA-256(`executionPolicySha256`)이 각 attempt의 시작·요청 예약·결과 기록에 남아 있어 다른 유효한 값으로 바뀌어도 탐지합니다. 재개 시 기록된 요청 수와 비용(정산 비용, 알 수 없으면 예약 비용)을 복원하고 0으로 되돌리지 않습니다. 예약 비용은 저장된 단가와 추정치로, 정산 비용은 저장된 usage와 단가로 기존 Judge 비용 규칙(`estimate_cost`)에 따라 다시 계산해 저장값과 같아야 하며, 계산할 수 없는 비용은 `null`이어야 합니다. 결과를 읽을 때(`require_completed`)도 Quiz가 여전히 Judge-ready이고 바뀌지 않았는지, 기록 연결과 hash, 저장된 모델 출력의 재검증 결과가 저장된 판정과 같은지 확인합니다.
- **정하지 않은 정책**: Quiz 전체 PASS/FAIL 집계와 Creator에게 보여 줄 방식, semantic `fail` 이후 재생성, Creator 승인, 게시. 이번에는 문항별 7개 항목 판정·reason, 실행 상태, provenance까지만 저장합니다.

## Production Judge 집계와 Creator 검토 상태 (Issue #43)

완료된 Production Pointwise 평가를 결정론적으로 집계해 Creator 검토용 요약과 routing을 제공합니다(`src/production_judge_aggregation.py`). Provider를 호출하지 않으며 Judge 원결과를 바꾸지 않습니다.

실행 상태(Judge evaluation state), semantic summary(판정 요약), review routing, Creator 승인은 서로 다른 값입니다.

| semantic summary | 조건 | review routing |
| --- | --- | --- |
| `ALL_PASS` | 모든 항목이 `pass` | `READY_FOR_CREATOR_REVIEW` |
| `HAS_FAIL` | `fail`이 하나 이상(`uncertain`도 함께 표시) | `ATTENTION_REQUIRED` |
| `UNCERTAIN_ONLY` | `fail` 없이 `uncertain`이 하나 이상 | `ATTENTION_REQUIRED` |
| `UNAVAILABLE` | 유효한 completed 평가 없음 | `JUDGE_UNAVAILABLE` |

- 어떤 routing도 승인이 아닙니다. `ALL_PASS`도 자동 승인·게시하지 않고, `HAS_FAIL`도 자동 폐기·재생성하지 않으며, 특정 항목을 필수 pass 조건으로 두지 않습니다. Judge는 `contentText` 기준 평가이며 원본 영상의 사실을 검증하지 않습니다.
- **집계 입력과 출력**: `ProductionJudgeRunner.require_completed()`로 검증된 completed 평가의 문항별 7개 항목 verdict·reason만 입력으로 씁니다. 출력은 문항 수, 항목 평가 수(문항 수 × 7), 전체·항목별·문항별 `pass`/`fail`/`uncertain` 개수, `fail`·`uncertain` 목록(`questionIndex`, rubric 항목, verdict, 원본 reason), 원본 문항별 verdict·reason 전체, semantic summary, review routing, 집계 정책 version(`production-judge-aggregation-v1`)과 digest입니다. 입력 순서와 관계없이 같은 판정은 같은 결과가 되며, 없는 verdict·reason을 만들지 않고 형식이 맞지 않는 입력은 거부합니다.
- **평가가 없을 때**: Judge 상태가 `not_started`, `retryable`, `exhausted`, `terminal`, `uncertain`이면 `UNAVAILABLE` / `JUDGE_UNAVAILABLE`과 실행 상태·사유를 읽기 전용으로 돌려주며 Creator 수동 검토를 허용합니다(`manualReviewAllowed`). 이 응답은 저장하지 않고, 실행 `uncertain`을 판정 `uncertain`으로 바꾸지 않습니다. `corrupted`와 `quiz_not_judge_ready`는 거부하며 수동 검토 경로로 돌리지 않습니다. `evaluation.json`이 일반 파일이 아니거나, `evaluation.json` 없이 평가 디렉터리에 다른 항목이 남아 있으면 `not_started`가 아니라 `corrupted`입니다. 예외는 생성이 끝나지 않은 create-once 임시 파일(`.production-` + 임의 8자 + `.tmp`, 일반 파일)뿐이며, 이것만 있으면 `not_started`로 이어서 실행할 수 있습니다.
- **저장**: 집계는 `results/production/operations/<quizOperationId>/judge/<evaluationId>/aggregations/<aggregationId>.json`에 한 번만 저장합니다. `aggregationId`는 Quiz provenance(operation, attempt, fingerprint, output·result SHA-256, Grounding run, `contentTextSha256`, snapshot SHA-256, validation version, prompt·출력 계약 version), Judge 설정(model, reasoning, `max_output_tokens`, prompt·rubric·schema version, 계약 hash)과 요청 본문 SHA-256, Judge evaluation(evaluation ID, attempt, fingerprint, 출력 SHA-256, 검증한 `result.json` 파일 바이트의 SHA-256), 집계 정책 version·digest의 SHA-256이며 `generatedAt`은 포함하지 않습니다. 같은 내용이라도 결과 파일 바이트가 다르면 다른 집계가 됩니다. 정책이 바뀌면 다른 파일이 되고 기존 파일은 그대로 남습니다.
- **재조회**: 저장된 집계는 Judge 평가를 다시 검증하고 집계를 다시 계산해 provenance와 결과가 모두 같을 때만 돌려줍니다. 저장값 변조, 형식 오류, 일부만 쓰인 파일, 읽기 실패, Quiz·Judge 결과가 바뀐 뒤의 오래된 집계는 거부하며 파일을 덮어쓰거나 고치지 않습니다. 집계 파일 쓰기가 실패하면(권한·디스크 오류 등) 거부하고 파일을 남기지 않으므로 다시 호출하면 됩니다. `verify_aggregation`은 같은 검사를 읽기 전용으로 수행하며, 집계 파일이 없으면 만들지 않고 거부합니다(Creator 상태 조회와 승인 gate가 사용). 호출자가 이미 계산해 비교한 기대값(`expected`)을 넘기면 다시 계산하지 않고 그 값과 정확히 같은 집계만 인정합니다.
- **후속 Creator 승인 요구사항**(이번 범위 밖): Creator 판단을 Judge 결과와 별도로 기록하고, Judge와 다르게 승인하면 override 여부와 사유를 필수로 남기며, Judge 원결과는 바꾸지 않고, Creator 승인 없이 게시하지 않습니다.

## Production Creator 검토·승인 결정 (Issue #45)

Creator가 Quiz와 Judge 결과(또는 Judge가 없다는 상태)를 함께 확인하고, Quiz operation 하나에 최종 승인 또는 거절을 한 번 기록합니다(`src/production_creator_review.py`). Judge 결과와 집계는 바꾸지 않으며, 게시는 후속 단계에서 승인 gate를 사용합니다. `decidedBy`는 호출자가 넘기는 이름이며 인증을 제공하지 않습니다.

- **검토 자료**(`review_packet(operationId, evaluationId)`): `require_judge_ready()`의 `contentText`와 문항(`questionIndex`, 질문, 보기, 정답 index, 설명, 근거), `review_status()`의 Judge 실행 상태·semantic summary·review routing·rubric별 판정과 reason·fail/uncertain 항목(`problemItems`)·집계 ID를 돌려줍니다. Judge 평가가 없으면 경고(`unavailableWarning`)를 붙입니다. Quiz eligibility 실패와 Judge `corrupted`·`quiz_not_judge_ready`는 거부합니다. 집계가 아직 없으면 `review_status()`가 집계 파일을 한 번 만듭니다.
- **결정 유형**: 승인 유형은 결정 시점의 routing에서 도출하며 호출자가 지정하지 않습니다.

| 결정 | routing | `decisionKind` | 필수 |
| --- | --- | --- | --- |
| `APPROVE` | `READY_FOR_CREATOR_REVIEW` | `JUDGE_AGREED` | 없음(사유는 선택) |
| `APPROVE` | `ATTENTION_REQUIRED` | `JUDGE_OVERRIDE` | 사유 + 모든 fail/uncertain 항목의 확인(`questionIndex`, rubric 집합이 집계와 정확히 일치) |
| `APPROVE` | `JUDGE_UNAVAILABLE` | `MANUAL_WITHOUT_JUDGE` | 사유. 당시 Judge 실행 상태를 기록 |
| `REJECT` | 모든 routing(`ALL_PASS` 포함) | `CREATOR_REJECTION` | 사유 |

- **저장**: `results/production/operations/<operationId>/creator-decision.json`에 한 번만 저장합니다(create-once, 원자적 저장). 결정, 결정 유형, `decidedBy`, `decidedAt`(UTC), 사유, 확인한 항목, 검토한 routing·summary·Judge 실행 상태·집계 ID·집계 정책, Quiz provenance(attempt, fingerprint, 출력 SHA-256, Grounding run, `contentTextSha256`, snapshot SHA-256, validation version), Judge evaluation ID와 attempt·fingerprint·출력·`result.json` SHA-256(Judge가 없으면 `null`)을 기록합니다.
- **최종 결정 1회**: 같은 요청(결정 시각 제외 모든 값이 같음)의 재전송은 저장된 기록을 돌려주고, 다른 결정·결정자·사유·확인 항목·검토 provenance는 거부합니다. 기록을 덮어쓰거나 철회·변경하지 않으며 손상된 기록도 새 결정으로 바꾸지 않습니다. 저장 실패는 거부하고 파일을 남기지 않습니다.
- **상태**(`decision_status`): `PENDING_CREATOR_REVIEW`(결정 없음), `CREATOR_APPROVED`·`CREATOR_REJECTED`(유효한 결정이고 검토한 Quiz·Judge·집계가 지금도 같음), `CREATOR_DECISION_STALE`(결정은 유효하지만 검토한 provenance가 바뀌었거나 더 이상 검증되지 않음), `CREATOR_DECISION_CORRUPTED`(결정 기록 자체가 잘못됨). 매번 upstream을 다시 검증하며 파일을 만들거나 바꾸지 않습니다. 결정 기록과 비교한 기대값 하나로 저장된 집계를 검증하고, 검증하는 동안 Judge 결과가 바뀌었으면(검증 끝에 다시 계산한 값이 처음과 다르면) 승인으로 인정하지 않습니다.
- **stale**: Quiz 변경·eligibility 실패, Judge `result.json` 변경·`corrupted`, 집계·집계 정책 변경, Judge 실행 상태 변경. `MANUAL_WITHOUT_JUDGE` 승인 뒤 같은 evaluation이 `completed`가 되면 stale이며, 결정은 한 번뿐이므로 같은 Quiz operation은 다시 승인할 수 없고 새 Quiz operation으로 다시 검토해야 합니다.
- **승인 gate**(`require_creator_approved(operationId)`): 상태가 `CREATOR_APPROVED`일 때만 `contentText`, 문항과 결정 기록을 돌려주고 그 밖에는 거부합니다. 파일을 만들거나 바꾸지 않으며 게시는 하지 않습니다.

## Production Quiz 로컬 발행 기록 (Issue #47)

Creator가 승인한 Production Quiz를 로컬 발행 기록으로 남기고, 그 기록의 무결성과 현재 유효성을 확인합니다(`src/production_publish.py`). 로컬 발행 기록은 승인된 Quiz와 그 승인 근거를 파일로 남긴 것이며, 실제 Cking 서비스에 Quiz가 공개됐다는 뜻이 아닙니다. 외부 서비스·DB·Provider를 호출하지 않습니다. `publishedBy`는 호출자가 넘기는 이름이며 인증을 제공하지 않습니다.

- **발행 가능 조건**(`record_publication(operationId, publishedBy)`): `ProductionCreatorReview.require_creator_approved()`가 승인한 Quiz만 기록합니다. 결정 없음, 거절, stale, 손상된 결정, Quiz eligibility 실패, Judge·집계 provenance 불일치는 그 gate에서 거부되며, Creator·Judge 정책을 이 모듈에서 다시 판단하지 않습니다.
- **저장**: `results/production/operations/<operationId>/publication.json`에 한 번만 저장합니다(create-once, 원자적 저장). 필드는 `format`, `publishId`, `policyVersion`(`production-local-publication-policy-v1`), `quizOperationId`, `creatorDecision`(gate가 검증한 결정 기록), `creatorDecisionSha256`(`creator-decision.json` 실제 파일 바이트의 SHA-256), `payload`(`contentText`와 문항), `payloadSha256`(payload의 canonical JSON SHA-256), `publishedBy`, `publishedAt`(UTC)입니다.
- **identity**: `publishId`는 policy version, operation ID, `creatorDecisionSha256`, `payloadSha256`의 canonical JSON SHA-256입니다. `publishedAt`은 포함하지 않습니다.
- **한 번만 기록**: `publishedAt`을 뺀 모든 값이 같은 요청은 저장된 기록을 그대로 돌려주고, 다른 `publishedBy`·승인·payload·policy는 거부합니다. 기존 기록은 손상된 경우에도 덮어쓰거나 고치지 않으며, 동시 요청에서도 기록은 하나만 남습니다. 쓰기 실패는 거부하고 파일을 남기지 않습니다.
- **저장 전후 검증**: gate 전후에 `creator-decision.json` 바이트가 같고 그 내용이 gate가 검증한 결정과 같아야 하며, 쓰기 직전에 바이트를 다시 확인합니다. 기록을 쓴 직후 현재 상태가 `LOCAL_PUBLICATION_CURRENT`가 아니면 정상 완료로 돌려주지 않고 거부합니다(`publicationRecordedButNotCurrent`). 이때 이미 쓴 기록은 지우지 않습니다. 여러 upstream 파일을 하나의 transaction으로 잠그지 않으므로 검증이 끝난 뒤의 변경까지 막지는 못합니다.
- **상태**(`publication_status(operationId)`): `LOCAL_PUBLICATION_NOT_RECORDED`(기록 없음), `LOCAL_PUBLICATION_CURRENT`(기록이 유효하고 지금도 같은 결정 파일·승인·payload가 검증됨), `LOCAL_PUBLICATION_STALE`(기록은 유효하지만 승인이나 Quiz가 지금은 다르거나 검증되지 않음), `LOCAL_PUBLICATION_CORRUPTED`(필드 누락·알 수 없는 필드·잘못된 형식·손상된 JSON·빈 파일·symlink·일반 파일이 아닌 경로·hash나 `publishId` 불일치·다른 operation, 또는 결정 파일 바이트는 같은데 snapshot이 다른 경우). 기록 자체의 검증은 upstream을 읽기 전에 하며, `payload`는 production Quiz 출력 계약(`evaluate_quiz_output`의 문항 수·필드·형식·근거 포함 규칙과 0부터 차례인 정수 `questionIndex`)을, `creatorDecision`은 Creator 결정 기록 계약을 정확한 JSON 형식까지 만족해야 합니다(`true`/`false`는 정수가 아님). 결정 기록의 Judge 실행 상태는 `completed`이거나 수동 검토를 허용하는 상태(`review_status()`의 `not_started`, `retryable`, `exhausted`, `terminal`, `uncertain`)여야 하며, `corrupted`·`quiz_not_judge_ready`·알 수 없는 값은 손상된 기록입니다. 결정 snapshot과 payload의 비교도 canonical JSON으로 해 형식이 다른 값을 같다고 보지 않습니다. 매번 다시 계산하며 파일을 만들거나 바꾸지 않습니다. 발행 이후 upstream이 바뀌어도 기록은 그대로 남고 상태만 stale이 됩니다.

## Production Quiz 검토용 번호 표시 (Issue #49)

Creator가 Quiz를 검토할 때 선택지와 정답을 1부터 매긴 번호로 볼 수 있는 읽기 전용 출력입니다(`src/production_quiz_display.py`). 저장된 `correctOptionIndex`와 `questionIndex`는 0부터 시작하는 그대로이며, 사람에게 보여 줄 때만 선택지는 1~4번, 정답은 `correctOptionIndex + 1`번으로 표시합니다. 파일을 만들거나 바꾸지 않고, Provider를 호출하지 않으며, 승인 결정이나 로컬 발행을 기록하지 않습니다.

```bash
python -m src.production_quiz_display --operation-id <quizOperationId>
```

```
문제 1. 다음 중 올바른 설명은?

1. 오답 A
2. 오답 B
3. 정답 C
4. 오답 D

정답: 3번 — 정답 C

해설: ...
근거: ...
```

- 명령은 `require_judge_ready()`로 Judge-ready Quiz만 읽습니다. `review_packet()`은 Judge 집계 파일을 만들 수 있으므로 이 명령에서 호출하지 않습니다. 코드에서는 `format_review_questions(packet)`에 `review_packet()` 결과를 그대로 넘겨도 같은 문항 표시를 얻습니다.
- 질문·선택지·해설·근거는 저장된 원문 그대로이며 순서를 바꾸지 않습니다. production Quiz 출력 계약을 만족하지 않는 문항(정답 index가 bool·실수·음수·범위 밖, 선택지 수·형식 오류 등)은 정답을 표시하지 않고 오류로 거부합니다.

## LLM-as-a-Judge 결과

A/B Quiz에 대한 사후 보조 평가(LLM-as-a-Judge) 결과는 `results/judge/<judgeRunId>/`에만 저장하며 위 세 JSONL과 `results/raw`, `results/evaluation`에는 쓰지 않습니다. 이 하위 디렉터리는 Pilot 저장 무결성 검사 대상이 아닙니다. Judge는 Pilot 결과와 Human Evaluation을 읽기만 합니다. 계약과 파일 구조는 [Judge 프로토콜](judge-protocol.md)을 따릅니다.

## 집계 CSV

세 평가를 각각 집계합니다. 모든 집계에는 `benchmarkType,method,model,promptVersion,runs,apiSuccessRate,latencyMeanMs,latencyP95Ms,meanInputTokens,meanOutputTokens,meanThinkingTokens,meanEstimatedCostUsd`와 적용 가능한 품질 지표를 포함합니다.

- Grounding: `factExistsPassRate,evidenceTypePassRate,timestampAccuracyPassRate,omissionFailRate,hallucinationFailRate`
- Quiz: `parseSuccessRate,validatorSuccessRate,beCompatibilityRate,answerAccuracyPassRate,uniqueAnswerPassRate,evidenceSupportsAnswerPassRate,koreanQualityPassRate`
- End-to-end: `parseSuccessRate,beCompatibilityRate,videoGroundingPassRate,answerAccuracyPassRate,hallucinationFailRate,meanTotalCostUsd`

각 비율은 **평가 가능한 승인 ground truth/질문 수**를 분모로 기록하고, `uncertain`·`null` 개수와 실행 불가 건수를 별도로 표시합니다. `gemini_direct_quiz`의 `beCompatibilityRate`는 빈 값입니다. 모델 품질을 비교하는 Quiz 집계는 같은 `contentTextSha256`과 프롬프트 버전의 실행만 묶습니다.

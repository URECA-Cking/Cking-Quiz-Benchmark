# 후속 실행 코드 경계

Pilot 구현 시 이 디렉터리에서 Video Grounding, 고정 `contentText` 기반 Quiz Generation, End-to-End 조합의 실행을 분리합니다. Gemini·DeepSeek 호출, 응답 파싱, 자동 평가, 지연 시간·토큰·비용 기록도 각각 구분합니다. Grounding과 Quiz 결과는 별도 JSONL에 쓰고 실행 ID로 연결합니다. 현재 실행 코드는 없습니다.

`Cking-BE`의 Provider 또는 Core Engine 코드를 복사해 독립 구현체로 발전시키지 않습니다. BE 계약을 평가 기준으로만 참조하고, Gemini 직접 Quiz 방식의 BE 호환성은 `not_applicable`로 기록합니다. 모델 비교에는 영상별 동일한 고정 `contentText`를 사용합니다. 외부 호출과 평가 결과는 운영 Quiz·Mission 데이터에 쓰지 않습니다.

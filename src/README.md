# 후속 실행 코드 경계

Pilot 구현 시 이 디렉터리에서 Gemini 호출, A/B/C 방식 실행, 응답 파싱, 자동 평가, 지연 시간·토큰·비용 기록을 분리합니다. 현재 실행 코드는 없습니다.

`Cking-BE`의 Provider 또는 Core Engine 코드를 복사해 독립 구현체로 발전시키지 않습니다. BE 계약을 평가 기준으로만 참조하고, 방식 B의 BE 호환성은 `not_applicable`로 기록합니다. 외부 호출과 평가 결과는 운영 Quiz·Mission 데이터에 쓰지 않습니다.

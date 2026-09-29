# Pilot 데이터 검토 상태

2026-09-29 기준 NASA Goddard/SVS 공식 페이지에 YouTube 링크와 영문 자막이 함께 있는 짧은 영상 세 편을 후보로 골랐습니다. 메타데이터는 `data/videos.jsonl`, 자막에 근거한 사실 후보는 `data/ground_truth_candidates.jsonl`에 있습니다. 후보는 AI가 공식 자막을 읽고 정리한 것이며 **사람이 영상을 직접 확인한 최종 ground truth가 아닙니다**. 모두 `pending_human`입니다.

| 영상 | 공식 페이지 | 자막 마지막 시각 | 사람 검수 시 중점 |
| --- | --- | --- | --- |
| Connect the Drops with NASA Data | https://svs.gsfc.nasa.gov/13254 | 3분 30초 | 물순환 관측과 모델 설명이 실제 영상에도 일치하는지 |
| Sources of Methane | https://svs.gsfc.nasa.gov/4799 | 1분 51초 | 지역별 배출원과 수치가 영상 발화에 정확히 있는지 |
| Largest Organics Yet Discovered on Mars | https://svs.gsfc.nasa.gov/14808 | 1분 29초 | 확인된 탐지 결과와 지방산 관련 가설을 구분하는지 |

영상 길이는 자막의 마지막 시각과 다를 수 있어 `durationSeconds`는 아직 기록하지 않았습니다. 공식 NASA 페이지의 YouTube 링크는 확인했지만 Gemini가 각 URL에 접근할 수 있는지는 실행 전 확인해야 합니다. 실제 영상이나 전체 자막을 이 공개 저장소에 복제하지 않습니다. NASA 미디어 이용 지침은 교육·정보 목적의 이용을 설명하면서 제3자 저작물에 별도 권리가 있을 수 있다고 명시합니다. 각 영상 페이지에도 음악 크레딧이 있으므로 링크·메타데이터·요약 사실만 공개합니다.

세 영상 모두 영어입니다. 이는 URL·근거 파이프라인의 첫 검증에는 적합하지만 **한국어 콘텐츠에서의 품질을 대표하지 못합니다**. 한국어 원문과 이용 허락이 확인된 크리에이터 소유 영상 한 편 이상을 추가한 뒤 언어별 결과를 분리해 평가해야 합니다. 사람이 원본 영상과 후보 사실·시점을 대조하고 `approved` 또는 `rejected`, 검수자와 검수 시각을 기록한 뒤에만 최종 ground truth로 사용합니다.

출처 및 이용 조건: NASA SVS 영상별 공식 페이지와 [NASA Images and Media Guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/).

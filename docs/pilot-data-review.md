# Pilot 데이터 검토 상태

2026-09-29 기준 NASA Goddard/SVS 공식 페이지에 YouTube 링크와 영문 자막이 함께 있는 영상 세 편, 한국항공우주연구원(KARI) 공식 페이지에 한국어 자막과 YouTube 링크가 있는 영상 한 편을 후보로 골랐습니다. 메타데이터는 `data/videos.jsonl`, 자막에 근거한 사실 후보는 `data/ground_truth_candidates.jsonl`에 있습니다. 이후 사용자가 YouTube 영상의 자막·내용과 후보 사실을 대조해 4개 영상의 12개 후보를 모두 맞는 것으로 판정했고, 현재 모두 `approved`입니다. 이는 원본 transcript 전문이나 timestamp의 프레임 단위 정확성을 검증했다는 뜻은 아닙니다.

| 영상 | 공식 페이지 | 길이 확인 근거 | 사람 검수 시 중점 |
| --- | --- | --- | --- |
| Connect the Drops with NASA Data | https://svs.gsfc.nasa.gov/13254 | 3분 30초 | 물순환 관측과 모델 설명이 실제 영상에도 일치하는지 |
| Sources of Methane | https://svs.gsfc.nasa.gov/4799 | 1분 51초 | 지역별 배출원과 수치가 영상 발화에 정확히 있는지 |
| Largest Organics Yet Discovered on Mars | https://svs.gsfc.nasa.gov/14808 | 1분 29초 | 확인된 탐지 결과와 지방산 관련 가설을 구분하는지 |
| 무중력환경시험설비 | https://www.kari.re.kr/kor/37/video/view/54 | YouTube 영상 3분 47초 | 110m·4.72초 수치, 비진공 시험 방식, 액체 거동 설명과 영상이 일치하는지 |

NASA 세 편의 영상 길이는 자막의 마지막 시각과 다를 수 있어 처음에는 `durationSeconds`를 기록하지 않았습니다. KARI 영상의 227초는 YouTube 페이지의 영상 메타데이터에서 확인했습니다. 이후 Grounding timestamp 검사를 보강하면서 2026-10-07에 NASA 세 편도 같은 YouTube 페이지 메타데이터로 기록했습니다(`approxDurationMs`를 초 단위로 올림한 값으로 페이지의 `duration` 표기와 같으며, KARI의 227초도 같은 규칙에 맞음): `nasa-water-cycle-2019` 211초(210.426초), `nasa-methane-2020` 119초(118.868초), `nasa-mars-organics-2025` 91초(90.023초). 세 값 모두 각 자막의 마지막 시각보다 큽니다. 공식 페이지들의 YouTube 링크는 확인했지만 Gemini가 각 URL에 접근할 수 있는지는 실행 전 확인해야 합니다. 실제 영상이나 전체 자막을 이 공개 저장소에 복제하지 않습니다. NASA 미디어 이용 지침은 교육·정보 목적의 이용을 설명하면서 제3자 저작물에 별도 권리가 있을 수 있다고 명시합니다. KARI 페이지는 공공누리 1유형으로 표시되어 있습니다. 링크·메타데이터·요약 사실만 공개합니다.

NASA 세 편은 영어, KARI 한 편은 한국어입니다. 언어별 결과를 분리해 평가합니다. KARI 공식 페이지의 구간 표시는 분 단위의 대략적 시작점이므로 후보 사실의 시각도 **정밀 타임스탬프가 아닌 검수 범위**입니다. YouTube 자동 생성 한국어 자막과 KARI 페이지 텍스트에 오인식·오탈자가 있을 수 있으므로 새 후보도 영상과 대조해야 합니다. 실제 Creator 소유 콘텐츠에 대한 일반화도 이 Pilot만으로 주장하지 않습니다. 새 후보는 사람이 영상과 후보 사실·시점을 대조하고 `approved` 또는 `rejected`, 검수자와 검수 시각을 기록한 뒤에만 ground truth로 사용합니다.

출처 및 이용 조건: NASA SVS 영상별 공식 페이지와 [NASA Images and Media Guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/), [KARI 공식 영상 페이지](https://www.kari.re.kr/kor/37/video/view/54)의 공공누리 표시.

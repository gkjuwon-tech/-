# Google DeepMind Genie 3 조사

> 조사일: 2026-09-25

## 한 줄 요약

텍스트(또는 이미지) 프롬프트로 **실시간으로 돌아다닐 수 있는 3D 세계**를 만들어 주는 DeepMind의 범용 "월드 모델". 영상 생성기와 달리 미리 만든 클립을 재생하지 않고, 사용자나 에이전트가 움직일 때마다 다음 프레임을 즉석에서 생성한다.

## 타임라인

| 시기 | 버전 | 핵심 |
|---|---|---|
| 2024-03 | Genie 1 | 2D 환경만, 약 1fps, 게임 영상 3만 시간으로 학습 |
| 2024-12 | Genie 2 | 3D 지원, 360p, 10~20초 정도 일관성 유지 |
| **2025-08-05** | **Genie 3** | 720p·24fps 실시간, 몇 분간 일관성, 연구 프리뷰(일부 연구자·크리에이터만) |
| 2026-01-29 | Project Genie | Google Labs 웹 프로토타입. 미국 AI Ultra 구독자(18세 이상)에게 공개 |
| 2026-02 | Waymo World Model | Waymo가 Genie 3를 기반으로 자율주행 시뮬레이터 구축 |

## Genie 3 기술 사양

- **해상도/속도**: 720p, 24fps, 실시간 반응
- **일관성**: 몇 분 동안 세계가 유지됨. 시각적 기억은 약 1분 전까지
- **생성 방식**: 자기회귀(autoregressive) 방식. 지금까지 지나온 궤적을 참고해 프레임을 하나씩 생성한다. NeRF나 Gaussian Splatting처럼 명시적인 3D 표현을 두지 않는데도 물체가 유지되는 성질(object permanence)이 **학습 과정에서 저절로 생겼다**
  - 예: 벽에 칠한 페인트가 시야에서 벗어났다 돌아와도 그대로 남아 있음
- **아키텍처**: 잠재 공간(latent space)에서 동작하는 시공간 트랜스포머이고, 파라미터는 약 110억 개로 알려져 있다
  - ⚠️ 파라미터 수는 DeepMind 공식 블로그에 없는 수치다. 위키백과 등 2차 자료에만 나온다
- **기반 기술**: Genie 2와 Veo 3(물리를 잘 이해하는 영상 생성 모델)의 연구를 이어받았다

## 주요 기능

1. **텍스트 → 인터랙티브 세계**: 자연 풍경·생태계(동물 행동, 식생), 물·빛·지형 같은 물리 현상, 역사적 장소, 판타지나 애니메이션 스타일 세계
2. **Promptable World Events**: 탐험하는 도중에 텍스트로 날씨를 바꾸거나 물체·캐릭터를 추가하는 식으로 세계를 수정할 수 있다
3. **에이전트 학습 환경**: DeepMind의 SIMA 에이전트를 Genie 3 세계에 넣어 목표 지향 작업을 수행시켰다. 이전 버전보다 긴 행동 시퀀스를 지원하고 "만약 ~라면" 같은 반사실(counterfactual) 시나리오도 실험할 수 있다
4. DeepMind는 Genie 3를 **AGI로 가는 디딤돌**로 소개한다. 로보틱스나 체화(embodied) AI 학습용 데이터를 무제한으로 만들어 낼 수 있다는 이유다

## DeepMind가 밝힌 한계

- 에이전트가 할 수 있는 행동의 범위가 제한적이다
- 여러 에이전트가 서로 상호작용하는 상황을 잘 모델링하지 못한다
- 실제 장소를 지리적으로 정확하게 재현하지 못한다
- 텍스트 렌더링이 약하다 (프롬프트에 텍스트를 넣어야 겨우 나온다)
- 상호작용은 몇 분까지만 가능하다 (몇 시간은 안 된다)

## Project Genie (2026-01-29, 일반 사용자용 프로토타입)

- **접근 조건**: Google AI Ultra 구독(월 $200)이 필요하고, 미국의 18세 이상 사용자만 쓸 수 있다. 다른 지역은 확대 예정이지만 일정은 발표되지 않았다
- **구성**
  - *World Sketching*: 환경과 캐릭터를 설명하면 **Nano Banana Pro**가 미리보기 이미지를 만들고, 사용자가 이를 다듬는다
  - *Exploration*: WASD·방향키·스페이스바로 조작하면 앞으로 갈 길이 실시간으로 생성된다. 720p 탐험 영상을 내려받을 수 있다
  - *Remixing*: 다른 사람의 월드 프롬프트를 고쳐 새 세계를 만들거나 갤러리를 둘러볼 수 있다
- **제약**: 한 세션은 **60초**까지다(자기회귀 방식이라 연산 비용이 크기 때문). 프롬프트대로 나오지 않는 경우가 있고, 캐릭터 조작이 늦게 반응하기도 하며, 물리가 부정확할 수 있다
- **콘텐츠 필터**: 선정적인 콘텐츠를 막고 일부 저작권 캐릭터(디즈니 등)도 차단한다. 다만 닌텐도 게임(마리오 64, 젤다 BotW)과 비슷한 세계는 만들어진 사례가 보도됐다
- **공개 API, 무료 티어, 오픈소스는 없다**

## 반응과 파급

- **평가**: The Verge는 "느리다", Fast Company는 "24fps는 실사용하기에 끊긴다"고 평했다
- **시장**: 출시 직후 Unity 등 게임 관련 기업의 주가가 떨어졌다는 Bloomberg 보도가 있다
- **Waymo World Model (2026-02)**: Genie 3를 사후 학습(post-training)시켜 카메라 영상과 **LiDAR 데이터**를 함께 생성하도록 만들었다. 토네이도, 침수된 도로, 도로 위 코끼리처럼 드문 상황을 시뮬레이션해 로보택시를 학습시킨다. 조작 방식은 주행 액션, 장면 레이아웃, 언어 세 가지다

## 시사점

- 월드 모델이 "영상 생성"에서 "상호작용하는 시뮬레이션"으로 넘어가고 있다. 게임, 로보틱스, 자율주행 학습 데이터 생성에 먼저 적용되고 있다
- 반면 60초 세션, 24fps, 720p, 고가 구독, API 부재 때문에 **아직은 실제 제품을 만드는 데 쓰기 어렵다.** 연구용·체험용 단계다

## 출처

- [Genie 3: A new frontier for world models — Google DeepMind (공식)](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
- [TechCrunch — DeepMind thinks Genie 3 is a stepping stone toward AGI](https://techcrunch.com/2025/08/05/deepmind-thinks-genie-3-world-model-presents-stepping-stone-towards-agi/)
- [9to5Google — Google rolling out 'Project Genie'](https://9to5google.com/2026/01/29/google-project-genie/)
- [Business Standard — Project Genie explained](https://www.business-standard.com/technology/tech-news/google-release-project-genie-what-is-it-explained-availability-126013000304_1.html)
- [Wikipedia — Genie (world model)](https://en.wikipedia.org/wiki/Genie_(world_model))
- [Wikipedia — Project Genie (website)](https://en.wikipedia.org/wiki/Project_Genie_(website))
- [Bloomberg — Waymo says Genie 3 simulations can help boost robotaxi rollout](https://www.bloomberg.com/news/articles/2026-02-06/waymo-says-genie-3-simulations-can-help-boost-robotaxi-rollout)
- [OpenCV — Genie 3: A New Frontier for World Models](https://opencv.org/blog/genie-3/)

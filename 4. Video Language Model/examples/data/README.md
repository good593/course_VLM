# 예제 영상 출처 및 라이선스

모두 다운로드/재사용이 자유로운 소스에서 받은 영상입니다. (수강생 재배포 목적 확인 완료)

| 파일 | 용도(노트북) | 출처 | 라이선스 |
|---|---|---|---|
| `1_scene_cuts_sweeta_commercial_1960s.mp4` | 1. Frame 추출/전처리 (장면전환 감지) | [Internet Archive – Prelinger Archives](https://archive.org/details/sweetacommer) — Sweeta Commercial (1960s, 58초) | Public Domain |
| `2_cooking_steps_buffet_supper_1946.mp4` | 2. 영상 요약 / 이벤트 탐지 | [Internet Archive – Prelinger Archives](https://archive.org/details/arranging_a_buffet_supper) — Arranging a Buffet Supper (1946, 약 6분) | Public Domain |
| `3_cctv_normal_camera_rotate.mp4` | 3. 행동 분석 (정상) | [Mixkit](https://mixkit.co/free-stock-video/security-camera-rotates-and-follow-the-object-48922/) | Mixkit Free License (무료 상업적 이용, 출처 표기 불필요) |
| `3_cctv_anomaly_two_thieves.mp4` | 3. 행동 분석 (이상행동) | [Mixkit](https://mixkit.co/free-stock-video/two-thieves-recorded-on-a-security-camera-31372/) | Mixkit Free License |
| `3_cctv_anomaly_thief_creep.mp4` | 3. 행동 분석 (이상행동, 보조) | [Mixkit](https://mixkit.co/free-stock-video/thief-creep-through-house-and-looks-up-at-camera-12830/) | Mixkit Free License |
| `4_long_video_uruguay_1949.mp4` | 4. Video RAG (긴 영상) | [Internet Archive – Prelinger Archives](https://archive.org/details/Uruguay1949) — Uruguay (1949, 약 18분 여행 다큐멘터리) | Public Domain |

## 선정 이유
- **1번**: 58초짜리 옛날 TV 광고로, 8~10회 정도의 컷 전환이 영상 전체에 고르게 분포 → 균등 샘플링 vs 장면전환 감지 비교에 적합 (히스토그램 차이 분석으로 사전 검증 완료)
- **2번**: 1946년 가정학(home economics) 교육 영상으로, 상차림 준비 과정이 여러 단계로 순서있게 진행됨 → 영상 요약·이벤트 탐지 실습에 적합
- **3번**: 보안 카메라 각도로 촬영된 "평상시" 클립(정상) + "침입자" 클립(이상행동) 조합 → 이상행동 탐지 실습에 적합
- **4번**: 우루과이 곳곳(수도/도시/전원/산업 등)을 순서대로 소개하는 18분 여행 다큐멘터리 → 주제가 뚜렷이 나뉘는 챕터 구조라 Video RAG(클립 분할 검색) 실습에 적합

## 참고
- Prelinger Archives(Internet Archive) 영상은 모두 Public Domain으로, 원본은 [archive.org/details/prelinger](https://archive.org/details/prelinger) 컬렉션에서 재검색 가능
- Mixkit 영상은 [Mixkit License](https://mixkit.co/license/) 기준 무료 상업적 이용 가능, 출처 표기 불필요 (단, Mixkit 소재 자체를 "스톡 영상 상품"으로 재판매하는 것은 금지)

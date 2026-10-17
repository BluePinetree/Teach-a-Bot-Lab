# 개발 계획

| 날짜 | 목표 | 완료 기준 (실제 실행으로 확인) |
|---|---|---|
| 10/17 | Phase 0 환경 조사, Phase 1 Joy-Con L/R 읽기·로깅 | 90초 테스트 요약 `checks` 전부 통과 |
| 10/18 | Phase 2 InputProvider(JoyCon/Keyboard/Replay), Phase 3 3D 엔진 선정·최소 장면, Phase 4 양손 조작 | 같은 조작 코드를 세 Provider가 공유 / 사람이 Joy-Con으로 분류 과제 15~30초 내 성공 |
| 10/19 | Phase 5 시범 기록·Replay 검증, Phase 6 k-NN BC, AI 자율 실행 | 저장 시범 Replay 재현 / AI가 현재 상태만 보고 행동 생성 |
| 10/20 | Phase 7 고정 Seed 평가, Phase 8 Takeover 교정, 통합 UI, 리허설 | 교정 전후 성공률을 같은 Seed로 비교(개선 여부와 무관하게 실제 수치) |
| 10/21 | 첫 견학 | |
| 10/22 | 장애 수정·안정화, 여유 시 양손 전달 게임 | |
| 10/23 | 두 번째 견학 | |

## 결정이 필요한 지점 (사용자 협의)
- Phase 1 결과에 따라 Joy-Con 라이브러리 유지/교체
- Phase 3 시작 시 3D 엔진: Ursina(Python 단일 프로세스) vs Unity. 기준: Windows 호환, 한글 경로, Joy-Con 동시 입력, 학습 코드 결합, 일정

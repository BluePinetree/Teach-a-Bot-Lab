# 진행 상황

## 마지막 완료 단계
**Phase 1 — Joy-Con L/R 연결 및 데이터 읽기 (2026-10-17 실기 검증 통과)**
Phase 0 — 환경 조사 (2026-10-17). 결과: `docs/ENVIRONMENT.md`

## Phase 1 검증 결과
실행: `tests/manual/test_joycon_connection.py` (화면 체크리스트 안내 버전), 사용자가 Joy-Con L/R을 직접 조작.
로그: `data/logs/joycon_20261017_211330.jsonl` (+ `.summary.json`, git 미추적)

| 통과 조건 | 판정 | 실측 근거 |
|---|---|---|
| L/R 식별, 바뀌지 않음 | 통과 | L=057E:2006, R=057E:2007. 재연결 후에도 L로 복귀 |
| 두 스틱 동시 입력 | 통과 | 0.1초 이내 양쪽 동시 편향. 범위 L x 605~3299 / y 1204~3388, R x 684~3416 / y 707~3036 (12-bit) |
| 버튼 누름·해제 | 통과 | L 11개, R 11개 전부 누름 감지, 종료 시 전부 해제 상태 |
| IMU 반응 | 통과 | 정지 시 \|a\| L 1.007G, R 1.094G / 흔들 때 자이로 최대 L 1094, R 1185 dps |
| 60초 연속 | 통과 | 175.8초, 프로그램 오류 없음 |
| 연결 해제 시 비정상 종료 없음 | 통과 | L 해제(118s) → 리더 스레드 OSError를 포착해 기록, R은 66.7Hz 유지 → L 재연결(172s) 후 0.1초 만에 데이터 재개 |
| 로그 저장 | 통과 | 리포트 단위 JSONL + 이벤트 + 요약 |

추가 실측:
- 리포트 66.7Hz(15ms), 리포트당 IMU 3샘플 → IMU 실효 200Hz. 리포트 간격 p99 ≈ 30ms.
- L과 R의 정지 시 z축 부호가 반대(L +1G, R −1G): 문서(dekuNukem)의 축 반전과 일치. Phase 2에서 좌표계 통일 필요.
- R의 정지 시 \|a\|가 1.09G로 L(1.01G)보다 높음 → Phase 2에서 정지 상태 재보정 검토.
- 스틱 중립값이 기기마다 다름(L ≈ (1940, 2348), R ≈ (2122, 1877)) → Phase 2 중립 보정 필수.
- 재연결 소요 54초 중 대부분은 L이 Windows HID 목록에 다시 나타나기 전(사용자가 버튼 누르기 전) 시간. 장치가 다시 보인 뒤에는 2초 주기 재시도로 즉시 연결됨. 1회 "read error"로 열기 실패 후 다음 시도에 성공.

## 현재 진행 단계
Phase 2 — 입력 추상화 계층 (착수 전)

## 미해결 문제
- Intel 내장 Bluetooth Error 상태 (Realtek USB 어댑터 사용 중, 건드리지 않음)
- 프로젝트 경로 한글 포함 → Phase 3 3D 엔진에서 확인
- joycon-python 라이선스 파일 확인 필요 (메타데이터 UNKNOWN)
- 재연결은 학생이 조이콘 버튼을 눌러야 시작됨 → 시연 가이드에 안내 문구 필요

## 다음 작업 3개
1. Phase 2: `InputProvider` 인터페이스 + `JoyConInputProvider`(스틱 중립 보정·Dead Zone·정규화 −1~1, ZL/ZR Debounce, L/R 좌표계 통일, 연결 해제 시 0 입력 안전 정지)
2. `KeyboardInputProvider`, `ReplayInputProvider` + 입력 기록/재생 형식(스키마 버전)
3. 단위 테스트(정규화·Dead Zone·Debounce) + 세 Provider 공통 인터페이스 확인 → Phase 3 엔진 선정 협의

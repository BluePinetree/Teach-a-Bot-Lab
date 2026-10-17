# 진행 상황

## 마지막 완료 단계
**Phase 2 — 입력 추상화 계층 (2026-10-17 실기 검증 통과)**
- Phase 1 — Joy-Con L/R 연결 및 데이터 읽기 (2026-10-17 통과, 커밋 633819b)
- Phase 0 — 환경 조사 (2026-10-17, `docs/ENVIRONMENT.md`)

## Phase 2 구현 내용
| 모듈 | 내용 |
|---|---|
| `input/base.py` | `InputFrame`/`HandInput` 표준 형식(schema v1), 고정 버튼 집합, 유효성 검사, `InputProvider`(connect/poll/disconnect) |
| `input/processing.py` | 스틱 중립 보정 + 방향별 범위 정규화, 원형 Dead Zone(0.12), Debounce(2회 연속 리포트 일치), 정지 시 자이로 바이어스 추정, R→L IMU 축 매핑 추정(Kabsch) |
| `input/joycon.py` | `JoyConInputProvider`: 백그라운드 연결 관리·재연결, 리포트 순서대로 처리(폴링 주기와 무관), 끊긴 손은 입력 0 |
| `input/keyboard.py` | `KeyboardInputProvider`: WASD+Q / 방향키+오른쪽 Ctrl (키 읽기 함수 주입 가능) |
| `input/replay.py` | `InputRecorder`(JSONL + 헤더), `load_recording`, `ReplayInputProvider`(스텝/실시간, 끝나면 안전 정지) |
| `input/reference_consumer.py` | 게임 조작 대역: 스틱으로 양손 X/Z 이동 + 집게 상태. Provider 간 동일성 검증용 |

## Phase 2 검증 결과
**자동 테스트** `pytest`: 34개 통과 (정규화·Dead Zone·Debounce·바이어스·축 매핑·프레임 왕복/검증·키보드·기록→재생 동일성)

**실기 테스트** `tests/manual/test_input_providers.py` (Joy-Con, 사용자 조작, 214초, 11,981프레임 @ 56Hz 폴링)
로그: `data/logs/input_joycon_20261017_213328.jsonl` (+ `.summary.json`, git 미추적)

| 항목 | 결과 | 근거 |
|---|---|---|
| 스틱 중립 보정 | 통과 | L (2028.5, 2287), R (2067, 1835). 이전 세션 L (1940→2009→2028)로 매번 달라짐 → 접속 시 보정이 필요함을 재확인 |
| 방향 규칙 (위=+y, 오른쪽=+x) | 통과 | 한 방향씩 지시, 양손 4방향 모두 ≥0.9, 반대 방향 입력 0회 |
| 놓으면 정확히 0 (Dead Zone) | 통과 | 양손 |
| 집게(ZL/ZR) | 통과 | Debounce 후 grip=True |
| 연결 해제 시 안전 정지 | 통과 | 끊긴 동안 해당 손 connected=False, stick (0,0), 버튼 전부 False (모든 프레임 유효성 검사 통과) |
| 재연결 | 통과 | L 164.4s 해제 → 210.6s 재연결 |
| 모든 프레임 스키마 유효 | 통과 | 무효 0 |
| 기록 → Replay 동일성 | 통과 | 기록 파일 재생 시 대역 소비자의 양손 궤적이 라이브와 완전히 일치 |
| 키보드 Provider | 통과 | 단위 테스트 + 3초 실행 |

**Phase 2 완료 조건**("동일한 게임 조작 인터페이스를 Joy-Con, 키보드, 기록 데이터가 공통 사용") 충족: 세 Provider 모두 같은 `InputFrame`을 내고, 같은 소비자 코드로 동작하며, Joy-Con 기록의 재생 결과가 라이브와 일치.

## 알려진 문제
- **R→L IMU 축 매핑 미검증.** 두 조이콘을 손으로 붙여 기울이는 방식은 실측 데이터에서 실패(같이 회전한 구간이 짧고, R 가속도 크기가 정지 시 1.09G로 9% 큼). 세션 중 상관계수 방식이 낸 (1,−1,−1)은 근거 불충분으로 폐기. 현재 R의 accel/gyro는 센서 좌표계 그대로이며 프레임 `meta.imu_frame_verified=false`. IMU는 MVP 필수가 아니므로 보류. 필요해지면 "책상 위 정지 자세 3가지"로 측정 (`scripts/estimate_axis_map_from_recording.py` 활용 가능).
- **재연결이 느림**: 해제 후 재연결까지 약 46~54초(3회 관찰). 장치가 Windows HID 목록에 다시 나타나기 전 시간이 대부분이며, 사용자가 버튼을 언제 눌렀는지는 기록되지 않아 원인(조이콘/Windows/코드) 미확정. 시연 중 연결이 끊기면 키보드 대체 입력으로 진행하는 운영 대책 필요.
- Intel 내장 Bluetooth Error 상태 (Realtek USB 사용 중)
- 프로젝트 경로 한글 포함 → Phase 3 엔진에서 확인
- joycon-python 라이선스 파일 확인 필요

## 현재 진행 단계
Phase 3 — 최소 3D 환경 (착수 전, 엔진 선택 협의 필요)

## 다음 작업 3개
1. 3D 엔진 결정: Ursina 설치·한글 경로·Joy-Con Provider와 같은 프로세스에서 60fps 동작 확인 → 사용자와 협의
2. 최소 장면(테이블, 좌우 집게, 큐브, 목표 상자, 고정 카메라) + `InputProvider`로 양손 X/Z 이동·ZL/ZR 집기
3. 재연결 지연 원인 확인용 짧은 측정(버튼 누른 시각 기록)

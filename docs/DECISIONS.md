# 기술 결정 기록

## D1. 프로젝트 Python 3.12 + uv 가상환경 (2026-10-17)
- 시스템에는 Anaconda Python 3.11만 있음. 요구사항(3.12)과 "기존 환경 수정 금지"를 지키기 위해 uv가 관리하는 3.12.9로 `.venv`를 만든다.

## D2. Joy-Con 입력: joycon-python + hidapi로 시작 (2026-10-17, Phase 1 검증 대기)
- 근거: Python 학습 코드와 같은 프로세스에서 쓸 수 있고, 의존성이 작으며(hidapi는 cp312 Windows 휠 존재), L/R을 PID(057E:2006 / 057E:2007)로 구분한다.
- 소스 확인으로 알게 된 제약과 대응 (`src/teachabot/input/joycon_device.py`):
  - get_status()는 리포트당 IMU 샘플 3개 중 첫 번째만 줌 → 3개 모두 직접 읽음.
  - 생성자가 타임아웃 없는 블로킹 read를 함 → 별도 스레드 + 타임아웃으로 감쌈.
  - 리더 스레드 종료/close API가 없음 → 재연결 시 세대(generation) 번호로 옛 핸들의 샘플을 버림.
  - 스틱 보정값을 쓰지 않음(원시 12-bit) → Phase 2에서 중립 보정·Dead Zone 구현.
  - serial을 무시하고 VID/PID로 첫 장치를 염 → L 1개 + R 1개 전제.
- **확정 (2026-10-17 Phase 1 실기 검증)**: Windows 11 + Realtek BT 5.0에서 L/R 동시 66.7Hz 리포트(IMU 200Hz), 버튼·스틱·IMU 전부 정상, 연결 해제·재연결 동작 확인. 라이브러리 교체 불필요.
- 대안(실패 시) — 현재 불필요: hidapi로 직접 프로토콜 구현(dekuNukem 문서), everything-imu(Rust, MIT, Joy-Con 200Hz 검증됨) 브리지, Unity + UnityJoyCon.

## D3. 입력 표준 형식과 신호 처리 (2026-10-17, Phase 2)
- `InputFrame` schema v1: 스틱 [-1,1] (+x 오른쪽, +y 위), 손별 고정 버튼 집합(grip=ZL/ZR), accel[G]/gyro[dps]. 끊긴 손은 connected=False + 중립값을 스키마 수준에서 강제 → 게임은 별도 처리 없이 안전 정지.
- 스틱: 접속 시 정지 상태 20리포트 중앙값으로 중립 보정(실측 중립이 세션마다 수십~수백 단위 이동), 방향별 범위를 관측 최대치로 확장(상한 1700), 원형 Dead Zone 0.12(축별 Dead Zone은 대각선 방향을 왜곡).
- Debounce 12ms = 15ms 리포트 기준 2회 연속 일치. 지연 약 15ms 추가.
- 신호 처리는 poll()에서 리포트 타임스탬프 순으로 수행 → 게임 프레임레이트와 입력 샘플링 분리.
- 기록 형식: JSONL(헤더 + 프레임), schema 이름·버전 검사. Replay는 학습 결과가 아님을 모듈 문서에 명시.

## D4. R→L IMU 축 매핑 보류 (2026-10-17)
- 손으로 붙여 기울이는 방식은 실측에서 신뢰할 수 없었음(동시 회전 구간 짧음, R 가속도 스케일 +9%). 상관계수 기반 추정은 결합된 축(x·z)에 속아 잘못된 확신을 냄 → Kabsch(최적 회전) 기반으로 교체하고 잔차·부호행렬 조건으로 검증.
- IMU는 MVP 조작에 쓰지 않으므로 축 통일은 필요할 때 정지 자세 방식으로 측정. 그 전까지 R IMU는 센서 좌표계, `meta.imu_frame_verified=false`.

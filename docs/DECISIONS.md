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
- 대안(실패 시): hidapi로 직접 프로토콜 구현(dekuNukem 문서), everything-imu(Rust, MIT, Joy-Con 200Hz 검증됨) 브리지, Unity + UnityJoyCon.

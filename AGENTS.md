# AGENTS.md — Teach-a-Bot Lab

## 세션 시작 시 먼저 읽을 것
1. `docs/PROJECT_CONTEXT.md` (요구사항 요약과 절대 규칙)
2. `docs/PROGRESS.md` (어디까지 검증됐는지, 다음 작업)
3. `docs/DECISIONS.md` (왜 이 스택인지)
4. `docs/ENVIRONMENT.md` (PC 환경)

## 핵심 목표
Joy-Con L/R로 3D 양손 집게를 조종 → 학생 시범 기록 → 실제 모방학습 → AI 자율 실행·평가 → 교정학습으로 개선. 연출·궤적 재생 금지.

## 개발 원칙
- Bottom-up: 현재 Phase가 실기 검증되기 전에 상위 Phase 코드를 대량 작성하지 않는다.
- Phase 완료마다 보고: 구현 기능 / 실행한 검증 / 성공·실패 / 알려진 문제 / 변경 파일 / 다음 목표. PROGRESS.md 갱신 후 코드와 문서를 함께 커밋.
- 입력 장치·시뮬레이션·학습·UI 모듈 분리. 게임 코드는 Joy-Con을 직접 참조하지 않는다.

## 현재 스택
Python 3.12 (`.venv`, uv), joycon-python 0.2.4 + hidapi 0.15.0, numpy, pytest. **현재 메인 환경(D10): MuJoCo 3.16 + Menagerie ALOHA 2 + mink IK** — `teachabot.aloha`(scene/env/teleop), 앱 `teachabot.aloha_app`. 모델 파일은 `scripts/download_aloha.py`로 받음(git 미추적). 한글 경로 때문에 MuJoCo에는 파일 경로 대신 바이트를 넘긴다. 입력은 `teachabot.input` (InputProvider: JoyCon/Keyboard/Replay). 3D: Ursina 8.3 (`teachabot.app`), 시뮬레이션 `teachabot.simulation`(엔진 독립), 화면 `teachabot.ui`, 모션 조작 `teachabot.input.motion`.

## 명령
```powershell
.venv\Scripts\python.exe scripts\list_hid_devices.py
.venv\Scripts\python.exe tests\manual\test_joycon_connection.py   # Phase 1 장치 테스트
.venv\Scripts\python.exe tests\manual\test_input_providers.py     # Phase 2 표준 입력 테스트 (--source keyboard 가능)
.venv\Scripts\python.exe -m pytest -q                             # 자동 테스트
.venv\Scripts\python.exe -m teachabot.app [--control tilt|point] [--input keyboard] [--check]  # 3D 앱
.venv\Scripts\python.exe scripts\check_ursina_env.py --input joycon  # 엔진·한글 경로·fps 점검
```
```powershell
.venv\Scripts\python.exe scripts\download_aloha.py                    # ALOHA 모델 (16MB, 고정 커밋 + sha256)
.venv\Scripts\python.exe -m teachabot.aloha_app [--check] [--input keyboard] [--move stick|tilt|point] [--wrist imu|off]
.venv\Scripts\python.exe -m teachabot.aloha_app --screenshot assets\_check\x.png   # 창 없이 화면 저장
```
```powershell
.venv\Scripts\python.exe -m teachabot.aloha_app --record s01             # 시범 기록 (익명 세션 코드)
.venv\Scripts\python.exe scripts\train_bc.py [--limit k | --curve 1 2 3 5 10]   # 모방학습 + 자동 평가(처음 보는 배치 20판)
.venv\Scripts\python.exe -m teachabot.aloha_app --policy data\policies\X.npz   # AI가 조종
.venv\Scripts\python.exe scripts\train_rl.py --reward sparse|dense       # 강화학습 기준선 (Ctrl-C로 중단해도 저장)
```
학습 데이터·정책·강화학습 결과는 `data/demos`, `data/policies`, `data/rl_runs` (git 미추적). 강화학습 실행 중에도 앱 프레임은 ~25ms로 문제없음.
ALOHA 앱은 두 조이콘이 연결되고 오른쪽 A(키보드 Space)를 눌러야 시작한다. Esc 또는 Ctrl-C로 끝내면 요약 JSON이 `data/logs`에 저장된다.
수동 테스트는 화면에 체크리스트가 뜨므로, 사용자에게 보이는 창(터미널 패널 등)에서 실행해야 한다.
3D 앱 화면을 직접 확인할 때는 `--screenshot PATH`(창을 맨 앞에 띄워 2초 뒤 캡처 후 종료)를 쓴다. Panda 버퍼 캡처는 검게 나온다.
조이콘을 쓰는 프로세스는 한 번에 하나만 띄운다(둘이면 HID read error).

## 하지 말 것
- 실제 실행 없이 테스트 성공 보고
- 성능 향상 연출, 시범 궤적을 AI 결과처럼 재생
- 드라이버 설치·삭제, 보안 설정 변경, 시스템 Python 수정 (사용자 승인 없이)
- 정식 DAgger 구현이라 주장

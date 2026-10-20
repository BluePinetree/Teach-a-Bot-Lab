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
- 커밋 메시지는 **Conventional Commits** 형식(2026-10-20 사용자 요청, `bc8fd28` 이전 커밋은 예전 형식):
  - 제목: `<type>(<scope>): <설명>` — 72자 이내, 명령형, 끝에 마침표 없음. 설명은 영어(기존 기록과 맞춤).
  - type: `feat`(새 기능) `fix`(버그 수정) `docs`(문서만) `test`(테스트만) `refactor`(동작 그대로 구조 변경) `perf`(성능) `build`(의존성·빌드) `chore`(그 밖의 잡무). 하위 호환을 깨면 `feat!:` 처럼 `!`를 붙이고 본문에 `BREAKING CHANGE:` 를 쓴다.
  - scope 예: `btlink`, `input`, `aloha`, `learning`, `app`, `scripts`. 여러 곳이면 생략한다.
  - 본문: 빈 줄 다음에 `- ` 목록으로 무엇을 왜 바꿨는지, 실측 숫자와 검증 결과(테스트 수, 실기 결과)를 적는다. 하지 않은 것·확인하지 못한 것도 적는다.
  - 한 커밋에는 한 종류의 변경만 담는다. 기능과 그 기능의 문서·테스트는 함께 묶어도 된다.
  - 예: `feat(btlink): auto re-pair stuck Joy-Cons and warn before the 15-minute sleep`
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
.venv\Scripts\python.exe -m teachabot.aloha_app --policy data\policies\X.npz [--record t01] [--video V.mp4]   # AI가 조종, 스틱/버튼으로 손별 넘겨받기, X(오른쪽)/T = 다시 학습
.venv\Scripts\python.exe scripts\render_rollout_video.py --policy X.npz --seeds 1000000 1000001 --out V.mp4 --label "..."   # 창 없이 정책 실행 녹화
.venv\Scripts\python.exe scripts\make_learning_figures.py              # fig13(시범 수별)·fig14(강화학습)·fig16(넘겨받기 효과) 갱신
.venv\Scripts\python.exe scripts\compare_datasets.py --seeds 0 1 2 --episodes 100   # 시범만 vs 시범+개입, seed별 공정 비교
.venv\Scripts\python.exe scripts\per_hand_stats.py data\policies\X.npz    # 손별 상자 안 / 들어 올림 / 멈춤
.venv\Scripts\python.exe scripts\train_rl.py --reward sparse|dense       # 강화학습 기준선 (Ctrl-C로 중단해도 저장)
.venv\Scripts\python.exe -m teachabot.aloha_app --visit 1조 --visit-demos 3 [--visit-base s01 t01] [--video V.mp4]   # 견학 모드 (docs/VISIT_RUNBOOK.md)
.venv\Scripts\python.exe scripts\make_visit_reference.py                # 견학 결과 화면 참고 숫자 갱신 (측정 파일에서)
.venv\Scripts\python.exe -m teachabot.aloha_app --task s2 [--record s2_01]   # S2 양팔 트레이 들기 (조작·시범 기록만, D16)
.venv\Scripts\python.exe -m teachabot.aloha_app ... [--ui classic]   # 기본은 패널 화면 1920x1080 (F 전체 화면, 오른쪽 Y/키보드 L 학습 화면), classic = 예전 1280x720 화면
```
학습 데이터·정책·강화학습 결과·영상은 `data/demos`, `data/policies`, `data/rl_runs`, `data/videos` (git 미추적). 배치 seed: 평가 1,000,000~ / 튜닝 2,000,000~ / AI 시연·개입 3,000,000~ (섞지 말 것). 강화학습 실행 중에도 앱 프레임은 ~25ms로 문제없음.
ALOHA 앱은 두 조이콘이 연결되고 오른쪽 A(키보드 Space)를 눌러야 시작한다. Esc 또는 Ctrl-C로 끝내면 요약 JSON이 `data/logs`에 저장된다.
수동 테스트는 화면에 체크리스트가 뜨므로, 사용자에게 보이는 창(터미널 패널 등)에서 실행해야 한다.
3D 앱 화면을 직접 확인할 때는 `--screenshot PATH`(창을 맨 앞에 띄워 2초 뒤 캡처 후 종료)를 쓴다. Panda 버퍼 캡처는 검게 나온다.
조이콘을 쓰는 프로세스는 한 번에 하나만 띄운다(둘이면 HID read error). 단, 연결 관리자(`scripts/joycon_manager.py`)는 앱이 떠 있는 동안 HID를 열지 않으므로 함께 띄워도 된다.
조이콘은 마지막 버튼 입력 15분 뒤 스스로 잠들고, 깨우면 자주 "고착"(페어링됨인데 HID가 죽음)되어 재페어링이 필요하다(`docs/JOYCON_LINK_MANAGER_PLAN.md` 9·10절). 실기 테스트 전에는 관리자를 먼저 띄우거나 `scripts/joycon_pair.py LR`로 재페어링한다. 진단 기록기: `scripts/joycon_link_log.py [--hold-hid]`.

## 하지 말 것
- 실제 실행 없이 테스트 성공 보고
- 성능 향상 연출, 시범 궤적을 AI 결과처럼 재생
- 드라이버 설치·삭제, 보안 설정 변경, 시스템 Python 수정 (사용자 승인 없이)
- 정식 DAgger 구현이라 주장
- 견학에서 `--test-autopilot` 사용 (시험 전용, 세션 코드 `_`로 시작, 실행 후 데이터 삭제)

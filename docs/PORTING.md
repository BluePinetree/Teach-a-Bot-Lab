# 새 PC로 옮기기 — AI 에이전트용 지시문

이 파일은 **옮겨 간 PC에서 일하는 AI 에이전트**(예: Claude Code)가 처음부터 끝까지 따라 하도록 쓴 것이다. 사람이 읽어도 된다.
목표: 압축을 푼 Teach-a-Bot Lab을 이 PC에서 견학 모드까지 실제로 돌아가게 만들고, 확인한 결과를 사용자에게 보고한다.

---

## 0. 먼저 읽고 지킬 것 (절대 규칙)
1. **사용자 승인 없이 하지 않는다**
   - 시스템 드라이버 설치·삭제
   - 보안·시스템 설정 변경. Bluetooth 페어링 해제·추가, 실행 정책 변경, 방화벽이 여기에 든다.
   - 프로그램 설치. uv와 Python 설치도 승인을 받는다.
   - 인터넷에서 파일 내려받기: 파일 이름·출처·크기를 말하고 승인을 받는다.
2. **기존 Python 환경(시스템 Python, Anaconda 등)은 건드리지 않는다.** 프로젝트 폴더의 `.venv`만 쓴다.
3. **실제로 실행해서 확인하지 않은 것은 성공했다고 보고하지 않는다.** 실패는 출력 그대로 보고한다.
4. 성능 향상을 연출하거나 녹화된 궤적을 AI 결과처럼 보여 주는 기능을 만들지 않는다.
5. 사용자 화면에 창을 띄우기 전에는 무엇이 뜨는지 먼저 말한다. 시험용 실행은 가능하면 창 없이 한다(`--screenshot`, `--test-autopilot`).
6. 커밋을 한다면 Conventional Commits(`feat(app): ...`, 영어 제목, `- ` 목록 본문)를 쓴다. `AGENTS.md`에 형식이 있다.
7. 프로젝트 규칙 전체는 `AGENTS.md` → `docs/PROJECT_CONTEXT.md` → `docs/PROGRESS.md` 순서로 읽는다.

## 1. 이 묶음에 든 것 / 빠진 것
| 들어 있음 | 설명 |
|---|---|
| 코드 전체 + `.git` | 커밋 기록 그대로. `git log --oneline | head`로 확인 |
| `assets/third_party/mujoco_menagerie/aloha/` | ALOHA 2 로봇 모델(BSD-3, 고정 커밋). 인터넷 없이 동작 |
| `config/joycon_link.json` | 이 조이콘 두 개의 MAC 주소 (L `XX:XX:XX:XX:XX:XX`, R `YY:YY:YY:YY:YY:YY`) |
| `config/motion_calibration.json` | 손목 기울기 보정값. **왼쪽 앞뒤 부호를 2026-10-20에 사용자 확인으로 뒤집은 상태** |
| `config/visit_reference.json` | 결과 화면 참고 숫자(출처 포함) |
| `data/demos/s1/` | 시범 데이터. 견학 바탕은 `s01`의 처음 5개(`--visit-base s01:5`). rehearsal* = 리허설 기록 |
| `data/policies/` | 학습된 모방학습 정책들(.npz + 시험 결과 .json) |
| `data/rl_runs/dense_20261019_184623/ckpt_11001856.zip` (+ `vecnorm_11001856.pkl`) | **강화학습 306시간 체크포인트.** 견학의 "AI 구경"과 비교 화면에 꼭 필요 |

| 빠져 있음 | 이유 / 대처 |
|---|---|
| `.venv` | PC마다 다시 만든다(2절). 다른 PC의 `.venv`는 동작하지 않는다 |
| `data/videos`, `data/logs` | 용량이 크고(수 GB) 실행에 필요 없다 |
| 나머지 강화학습 체크포인트 | 실행에 필요 없다(필요하면 원래 PC에서 따로 복사) |
| `models/hand_landmarker.task` | 보류된 웹캠 기능용. 필요하면 `scripts/download_models.ps1`(인터넷, 승인 필요) |
| 맑은 고딕 글꼴 | Windows 기본 글꼴(`C:\Windows\Fonts\malgun.ttf`, `malgunbd.ttf`)을 쓴다. 저장소에 복사하지 않는다 |

## 2. 설치 (Windows 10/11)
압축을 푼 폴더(예: `...\Teach-a-Bot-Lab`)에서 PowerShell로 진행한다. **경로에 한글이 있어도 된다**(MuJoCo에는 파일 경로 대신 바이트를 넘기도록 되어 있음).

1. 확인만 하는 명령(승인 불필요)
   ```powershell
   uv --version
   git --version
   Test-Path C:\Windows\Fonts\malgun.ttf
   Test-Path C:\Windows\Fonts\malgunbd.ttf
   ```
2. `uv`가 없으면 **사용자에게 설치 승인을 받는다**. 공식 설치 안내: https://docs.astral.sh/uv/getting-started/installation/
   - 방법 예: `winget install --id=astral-sh.uv -e`
   - 승인이 없으면 여기서 멈추고 보고한다.
3. 프로젝트 전용 가상환경을 만든다. Python 3.12를 uv가 내려받을 수 있으니, 처음이면 내려받기라고 먼저 알린다.
   ```powershell
   uv venv --python 3.12 .venv
   uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
   ```
   - 패키지 버전은 `pyproject.toml`에 모두 고정되어 있다. torch·mujoco 등으로 수백 MB를 내려받는다.
4. 로봇 모델 파일이 온전한지 확인한다. 파일이 있으면 sha256만 검사하고 내려받지 않는다. 문제가 있을 때만 인터넷을 쓰니, 그 경우 먼저 알린다.
   ```powershell
   .venv\Scripts\python.exe scripts\download_aloha.py
   ```
   "done"이 나와야 한다.

## 3. 창 없이 하는 확인 (승인 불필요, 사용자 화면에 아무것도 안 뜸)
| # | 명령 | 기대 결과 |
|---|---|---|
| 1 | `.venv\Scripts\python.exe -m pytest -q` | **121 passed** (2026-10-20 원래 PC 기준). 숫자가 다르면 그대로 보고 |
| 2 | `.venv\Scripts\python.exe -m teachabot.aloha_app --screenshot assets\_check\port_s1.png` | "saved ..." 그리고 PNG에 로봇·큐브·패널 화면 |
| 3 | `.venv\Scripts\python.exe -m teachabot.aloha_app --task s2 --screenshot assets\_check\port_s2.png` | 트레이·공 장면 |
| 4 | 견학 전체 자동 시험(시험 전용):<br>`.venv\Scripts\python.exe -m teachabot.aloha_app --visit _porttest --visit-demos 2 --visit-base s01:5 --test-autopilot --max-seconds 500` | 오류 없이 끝나고 `data\logs\aloha_s1_visit_*.json` 생성. 그 안 `visit.steps` = intro → practice → demo → train → compare1 → ai → retrain → compare2 → result. `frame_ms_p50`을 기록(원래 PC 37 ms) |
| 5 | 4번이 남긴 시험 데이터 지우기(규칙) | `data\demos\s1\_porttest_*` 와 그 실행 중에 생긴 `data\policies\live_<그 시각>_*` 만 삭제. **다른 파일은 지우지 않는다** |

- MuJoCo 화면 오류(OpenGL/GLFW): 그래픽 드라이버 문제다. 드라이버 설치·변경은 **승인을 받아야 한다**. 원래 PC는 NVIDIA RTX A6000이다.
- 콘솔에 한글이 깨지거나 `UnicodeEncodeError`가 나면 `$env:PYTHONIOENCODING="utf-8"`로 다시 실행한다.

## 4. 조이콘 연결 (사용자와 함께)
- 조이콘(Switch 1 L/R)은 Bluetooth로 이 PC에 **새로 페어링**해야 한다. 페어링은 시스템 설정 변경이므로 **사용자 승인 후** 진행한다.
  - 방법 A(사용자 직접): Windows 설정 → Bluetooth → 장치 추가. 조이콘 옆 SYNC 버튼을 길게 누른다.
  - 방법 B(스크립트, 승인 후): `.venv\Scripts\python.exe scripts\joycon_pair.py LR`. 기존 페어링을 지우고 다시 맺는다. 화면 안내대로 SYNC를 누른다. `config/joycon_link.json`의 MAC만 받아들인다.
- 확인: `.venv\Scripts\python.exe scripts\list_hid_devices.py` 결과에 Joy-Con (L)과 (R)이 보여야 한다.
- 연결 관리자: `JoyCon_Manager.bat`(더블클릭) 또는 `.venv\Scripts\python.exe scripts\joycon_manager.py`. 두 손이 "HID 확인 정상"이어야 한다.
  - 관리자는 앱이 떠 있는 동안 HID를 열지 않으니 함께 띄워도 된다.
  - **조이콘을 쓰는 다른 프로그램은 한 번에 하나만** 띄운다(둘이면 HID read error).
- 조이콘은 버튼 입력 없이 15분이 지나면 잠든다. 깨운 뒤 "고착"되면 관리자 안내대로 SYNC를 누르거나 다시 페어링한다(`docs/JOYCON_LINK_MANAGER_PLAN.md` 9·10절).
- Bluetooth 어댑터: 원래 PC는 Realtek USB 어댑터를 썼고 내장 Intel은 오류 상태였다. 이 PC의 상태는 확인만 하고 고치지는 않는다.

## 5. 화면에 띄워서 하는 확인 (띄우기 전에 사용자에게 알린다)
1. 자유 조종 15초 측정(키보드 입력, 저절로 닫힘):
   ```powershell
   .venv\Scripts\python.exe -m teachabot.aloha_app --input keyboard --autostart --max-seconds 15
   ```
   - 요약 JSON(`data\logs\aloha_s1_run_*.json`)에서 `frame_ms_p50`과 `window_show_ms`를 기록한다.
   - 원래 PC(3840×2160, 150%)에서는 24 ms / 15 ms였다.
2. 조이콘 조종 확인. 사용자가 직접 쥔다.
   ```powershell
   .venv\Scripts\python.exe -m teachabot.aloha_app
   ```
   - A로 시작한다.
   - 스틱 = 이동, ZL/ZR = 집기, L/R = 내리기.
   - 조이콘을 기울이면 손목이 따라온다. 확인할 것: **왼손·오른손 모두 앞으로 기울이면 집게도 앞으로 기운다.** 좌우 회전은 1.5배 감도다.
   - 방향이 반대면 `config/motion_calibration.json`의 해당 손 `forward_dir` 부호를 뒤집는다. 먼저 백업하고, 사용자 확인을 받은 뒤에 한다.
   - 처음부터 다시 보정하려면 `.venv\Scripts\python.exe -m teachabot.app` → C키 보정 → 위 확인을 다시 한다.
3. S2 확인: `.venv\Scripts\python.exe -m teachabot.aloha_app --task s2`
   - L/R을 누른 채 손목을 꺾으면 높이가 바뀐다. 앞쪽이 내려가면 내려간다.
4. 화면 설정 키
   - **F** = 전체 화면 전환.
   - 1080p TV에서는 확대가 없어서 가장 가볍다. 4K에서 느리면 `--sharp-3d off`를 쓴다. 문제가 생기면 `--ui classic`(예전 화면).
5. 견학 리허설: `docs/VISIT_RUNBOOK.md`의 명령을 쓴다.
   ```powershell
   .venv\Scripts\python.exe -m teachabot.aloha_app --visit 1조 --visit-demos 3 --visit-base s01:5 --video data\videos\visit_1조.mp4
   ```
   - 순서: AI 구경(A로 시작) → B 1초 → 연습 → B 1초 → 시범 3판 → X → 공부 → 비교 → B 1초 → AI 시연(넘겨받기) → X → 다시 공부 → 비교 → B 1초 → 결과 → Esc.
   - 오른쪽 Y = 학습 화면, 키보드 C = 3개 동시 비교.

## 6. 사용자에게 보고할 것 (형식)
- 설치: uv와 Python 버전, `.venv` 생성 여부, 설치 중 오류
- 창 없는 확인: pytest 통과 수, 스크린샷 2장 경로, 자동 시험 단계 목록과 `frame_ms_p50`, 지운 시험 데이터 목록
- 조이콘: 페어링 방법과 결과, 관리자 상태
- 화면 확인: 자유 조종 `frame_ms_p50` / `window_show_ms`, 화면 해상도와 배율, 손목 방향 확인 결과
- 하지 못한 것과 사용자 결정이 필요한 것(승인을 기다리는 항목 포함)

## 7. 자주 나는 문제
| 증상 | 원인 / 대처 |
|---|---|
| `.venv\Scripts\python.exe`가 안 됨 | 다른 PC에서 복사한 `.venv`. 지우고 2절을 다시 한다 |
| 조이콘 "연결 안 됨" | 잠들었거나 고착. 관리자 안내, 아니면 `scripts\joycon_pair.py LR`(승인) |
| HID read error | 조이콘을 여는 프로그램이 둘. 하나를 닫는다 |
| 글자가 네모(□) | 맑은 고딕에 없는 글자. 코드에서 그 글자를 바꾼다(예: ⓪, −는 쓰지 않음) |
| 화면이 흐림 | DPI 처리(`teachabot/display.py`)가 동작하는지, 창 크기와 `window_show_ms`를 확인 |
| 견학 "AI 구경"이 계속 "불러오는 중" | `data/rl_runs/dense_20261019_184623/ckpt_11001856.zip`과 `vecnorm_11001856.pkl`이 있는지 확인 |

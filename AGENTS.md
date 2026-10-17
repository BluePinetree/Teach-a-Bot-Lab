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
Python 3.12 (`.venv`, uv), joycon-python 0.2.4 + hidapi 0.15.0. 3D 엔진 미정(Ursina 우선 검토).

## 명령
```powershell
.venv\Scripts\python.exe scripts\list_hid_devices.py
.venv\Scripts\python.exe tests\manual\test_joycon_connection.py
```
수동 테스트는 화면에 체크리스트가 뜨므로, 사용자에게 보이는 창(터미널 패널 등)에서 실행해야 한다.

## 하지 말 것
- 실제 실행 없이 테스트 성공 보고
- 성능 향상 연출, 시범 궤적을 AI 결과처럼 재생
- 드라이버 설치·삭제, 보안 설정 변경, 시스템 Python 수정 (사용자 승인 없이)
- 정식 DAgger 구현이라 주장

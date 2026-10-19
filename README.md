# Teach-a-Bot Lab

고등학생 연구실 견학용 데모. Joy-Con L/R로 3D 로봇 양손 집게를 조종하고, 학생 시범으로 AI가 모방학습하는 과정을 실제 학습·평가 수치로 보여준다.

- 요구사항 요약: `docs/PROJECT_CONTEXT.md`
- 진행 상황: `docs/PROGRESS.md`
- 에이전트 작업 규칙: `AGENTS.md`

## 준비

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e .
```

## 조이콘 연결 관리자 (더블클릭)

`JoyCon_Manager.bat`을 더블클릭하면 작은 창이 항상 위에 뜬다(콘솔 창 없음, 한 번에 하나만 실행). 앱보다 먼저 켜 두고 계속 둔다. 앱과 함께 떠 있어도 충돌하지 않는다.

- 손마다 **준비됨 / 잠듦 / 연결 중 / 고착 / 재페어링 / 페어링 필요**와 "조이콘이 스스로 잠들기까지 남은 시간"(마지막 버튼 입력 후 15분)을 보여 준다.
- 화면에 "SYNC 버튼을 누르세요"가 뜨면 조이콘 레일 옆 작은 SYNC 버튼을 한 번 누른다. Windows 설정 화면은 열 필요가 없다.
- 고착(페어링됐는데 신호가 안 옴)은 자동으로 감지해 다시 페어링한다. 남은 시간이 3분 이하이면 앱 화면에도 노란 줄이 뜬다 → 아무 버튼이나 한 번 누른다.
- 끌 때는 창을 닫는다. 기록: `data/logs/link_manager_<시각>.log`. 자세한 근거: `docs/JOYCON_LINK_MANAGER_PLAN.md`

## 다른 PC로 옮기기

1. 폴더를 복사하거나 git으로 받는다. `.venv`는 PC마다 다시 만들어야 한다(복사해 온 `.venv`는 원래 PC의 파이썬을 가리킨다).
2. [uv](https://docs.astral.sh/uv/)를 설치하고 `JoyCon_Manager.bat`을 더블클릭한다. `.venv`가 없거나 동작하지 않으면 다시 설치할지 묻는다(Y → 몇 분). 직접 하려면 위 "준비"의 두 줄을 실행한다.
3. 관리자 창의 "SYNC 버튼을 누르세요"에 따라 L/R의 SYNC를 누르면 새 PC에 페어링된다. 원래 PC로 돌아가 붙지 않으면 그쪽 관리자에서도 SYNC를 누르면 된다(고착으로 보고 다시 페어링함). 두 PC를 오가는 경우는 아직 실기로 확인하지 않았다.
4. **다른 조이콘**을 쓸 때는 `config/joycon_link.json`을 지운다. 그 파일에 적힌 MAC 주소의 조이콘만 자동으로 받아들이기 때문이다. 지우면 처음 페어링하는 L/R의 주소를 다시 저장한다.
5. 앱에 필요한 나머지: ALOHA 모델 `scripts\download_aloha.py`, 학습된 정책·시범(`data/policies`, `data/demos`)은 git에 없으므로 필요하면 따로 복사한다.

## Joy-Con 확인 (Phase 1)

1. 위의 연결 관리자로 페어링한다(예전 방법: Windows 설정 → Bluetooth 및 장치 → 장치 추가 → SYNC 버튼으로 L/R 각각 페어링)
2. `.venv\Scripts\python.exe scripts\list_hid_devices.py` 로 `057E:2006`, `057E:2007` 확인
3. `.venv\Scripts\python.exe tests\manual\test_joycon_connection.py --duration 90`
   결과: `data/logs/joycon_<시각>.jsonl` (모든 리포트) + `.summary.json` (통과 판정)

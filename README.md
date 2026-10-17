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

## Joy-Con 확인 (Phase 1)

1. Windows 설정 → Bluetooth 및 장치 → 장치 추가 → Bluetooth, Joy-Con 레일의 SYNC 버튼을 길게 눌러 L/R 각각 페어링
2. `.venv\Scripts\python.exe scripts\list_hid_devices.py` 로 `057E:2006`, `057E:2007` 확인
3. `.venv\Scripts\python.exe tests\manual\test_joycon_connection.py --duration 90`
   결과: `data/logs/joycon_<시각>.jsonl` (모든 리포트) + `.summary.json` (통과 판정)

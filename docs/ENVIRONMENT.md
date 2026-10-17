# 개발 환경 (Phase 0 조사 결과)

조사일: 2026-10-17. 모든 항목은 실제 명령 실행 결과다.

| 항목 | 결과 | 비고 |
|---|---|---|
| OS | Windows 11 Pro 10.0.26200 (64-bit) | |
| 시스템 Python | 3.11.4 (Anaconda, `%USERPROFILE%\anaconda3`) | **수정하지 않음** |
| 프로젝트 Python | CPython 3.12.9 (uv 관리, `%APPDATA%\uv\python\...`) | 프로젝트 전용 `.venv` |
| uv | 0.8.20 | 가상환경·패키지 관리 |
| Git | 2.39.1.windows.1 | 저장소 `git init` 완료 |
| GPU | NVIDIA RTX A6000 48GB, 드라이버 595.79 | MLP 학습엔 과분, CPU로도 충분 |
| Bluetooth | Realtek Bluetooth 5.0 USB 어댑터: **정상** / Intel Wireless Bluetooth (내장): **Error 상태** | Realtek로 페어링. Intel 쪽 문제는 건드리지 않음 |
| Joy-Con 페어링 | 조사 시점 미페어링 (HID 열거 0개) | 사용자 페어링 후 재확인 |
| Unity | 미설치 (Hub/Editor 경로 없음) | 대안 엔진으로 쓰려면 설치 필요 |
| 프로젝트 폴더 | `%USERPROFILE%\Projects\잡동사니\Teach-a-Bot-Lab` (사용자 지정) | 경로에 한글 포함 → 3D 엔진(Panda3D/Ursina) 경로 처리 Phase 3에서 확인 |

## 설치한 의존성 (`.venv`)

| 패키지 | 버전 | 용도 | 라이선스 / 상태 |
|---|---|---|---|
| joycon-python (pyjoycon) | 0.2.4 | Joy-Con HID 프로토콜 (SPI 보정값 읽기, 6축 IMU 활성화, 0x30 리포트 파싱) | 메타데이터상 UNKNOWN, 저장소 LICENSE 파일 별도 확인 필요. README는 macOS 기준, Windows 명시 없음 |
| hidapi (cython-hidapi) | 0.15.0 (2025-12) | Windows HID 접근 | BSD / GPLv3 등 택1. cp312 win_amd64 휠 제공 |
| pyglm | 2.8.3 | pyjoycon 의존성 (자이로 트래킹 클래스용) | |

## 실행

```powershell
cd %USERPROFILE%\Projects\잡동사니\Teach-a-Bot-Lab
.venv\Scripts\python.exe scripts\list_hid_devices.py          # 페어링 확인
.venv\Scripts\python.exe tests\manual\test_joycon_connection.py --duration 90
```

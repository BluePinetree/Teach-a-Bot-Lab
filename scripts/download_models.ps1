# Downloads model files that are not committed to git.
# MediaPipe Hand Landmarker (float16), Google, Apache-2.0, ~7.8 MB
$dst = Join-Path $PSScriptRoot "..\models"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile (Join-Path $dst "hand_landmarker.task")

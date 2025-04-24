#!/bin/bash

# 스크립트 실행 디렉토리 (ensemble_multi_attak.py가 있는 Ai_proj 폴더의 상위 디렉토리)
SCRIPT_DIR="/home/work/AIprogramming"
PYTHON_SCRIPT="Ai_proj/ensemble_multi_attak.py"
LOG_FILE="Ai_proj/ensemble_output.log"
PYTHON_EXECUTABLE="/opt/conda/bin/python" # 사용자 환경의 Python 경로

# 스크립트 디렉토리로 이동 (상대 경로 의존성 해결)
cd "$SCRIPT_DIR" || { echo "Error: Could not change directory to $SCRIPT_DIR"; exit 1; }

echo "Starting ensemble multi-attack script in the background..."
echo "Python executable: $PYTHON_EXECUTABLE"
echo "Python script: $PYTHON_SCRIPT"
echo "Output will be logged to: $LOG_FILE"

# nohup을 사용하여 백그라운드 실행 및 로그 저장
nohup "$PYTHON_EXECUTABLE" "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &

# 백그라운드 프로세스 ID (PID) 출력 (선택 사항)
BG_PID=$!
echo "Script started in background with PID: $BG_PID"
echo "You can monitor the output using: tail -f $LOG_FILE"
echo "Check process status using: ps -p $BG_PID"

exit 0
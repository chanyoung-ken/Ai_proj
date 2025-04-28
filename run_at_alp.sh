#!/usr/bin/env bash

# ================================================
# run_at_alp.sh
# AT+ALP multi-attention CIFAR10 training in background
# ================================================

# (1) 프로젝트 디렉토리로 이동
cd /home/work/AIprogramming/Ai_proj   # ← 여기를 실제 경로로 바꿔주세요

# (2) 로그 폴더 생성
LOG_DIR="multi_ensem_logs"
mkdir -p "$LOG_DIR"

# (3) 실행 스크립트 & 로그 파일 설정
SCRIPT="realensemble.py"   # ← 실제 .py 파일명으로 바꿔주세요
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/at_alp_${TIMESTAMP}.log"

# (4) 백그라운드 실행 (nohup + 로그 리다이렉트)
nohup python3 "$SCRIPT" > "$LOG_FILE" 2>&1 &

# (5) PID와 로그 위치 출력
echo "▶ AT+ALP training started, PID=$!"
echo "▶ Logging to $LOG_FILE"

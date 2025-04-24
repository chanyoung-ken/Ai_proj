#!/usr/bin/env bash

# ============================================
# Pipeline runner for HPO and benchmark scripts
# ============================================

# Exit immediately on errors
set -e

# Directory to store logs
LOG_DIR=logs
mkdir -p "$LOG_DIR"

# Timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

# Start pipeline
echo "$(timestamp): Starting hyperparameter search..." | tee -a "$LOG_DIR/pipeline.log"
python3 hyperparameter_tuning.py >> "$LOG_DIR/pipeline.log" 2>&1

echo "$(timestamp): Hyperparameter search completed." | tee -a "$LOG_DIR/pipeline.log"

# Run benchmark
echo "$(timestamp): Starting benchmark..." | tee -a "$LOG_DIR/pipeline.log"
python3 benchmark.py >> "$LOG_DIR/pipeline.log" 2>&1

echo "$(timestamp): Benchmark completed." | tee -a "$LOG_DIR/pipeline.log"

# End of script

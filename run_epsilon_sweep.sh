#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# List of epsilon values to test (using float values)
# Example values: 4/255, 8/255, 12/255
# ε = [2/255,4/255,…,16/255]
EPSILON_VALUES=(0.00784 0.01568 0.02352 0.03137 0.03921 0.04705 0.05489 0.06275)

# Python script to run
PYTHON_SCRIPT="realensemble.py"

# Directory for individual run logs
LOG_DIR="epsilon_run_logs"
mkdir -p "${LOG_DIR}"

# Number of epochs for each run
EPOCHS=100 # Or you can set this per epsilon if needed

# --- MLflow Configuration ---
# Set the MLflow experiment name for all runs triggered by this script
export MLFLOW_EXPERIMENT_NAME="CIFAR10_Epsilon_Sweep"
echo "Using MLflow Experiment: ${MLFLOW_EXPERIMENT_NAME}"

# --- Run Experiments ---
echo "Starting epsilon sweep... MLflow will track results."

for eps in "${EPSILON_VALUES[@]}"; do
    echo "-------------------------------------------------"
    echo "Running experiment for epsilon = ${eps}"
    echo "-------------------------------------------------"
    
    log_file="${LOG_DIR}/run_eps_${eps//./_}.log"
    
    # Run the python script with MLflow enabled
    # Redirect stdout and stderr to log file, and also show on console via tee
    # realensemble.py will now automatically log parameters and metrics to MLflow
    python "${PYTHON_SCRIPT}" --adv_eps "${eps}" --epochs "${EPOCHS}" 2>&1 | tee "${log_file}"
    
    # Check the exit code of the python script (from the left side of the pipe)
    script_exit_code=${PIPESTATUS[0]}
    if [ ${script_exit_code} -ne 0 ]; then
        echo "Error: Python script for epsilon ${eps} failed with exit code ${script_exit_code}. Check log file: ${log_file}" >&2
        # Optionally continue to the next epsilon or exit the script
        # exit 1 # Uncomment to stop the script on error
        continue # Continue with the next epsilon
    fi
    
    # Results extraction is no longer needed here, MLflow handles it.
    # echo "Extracting results for epsilon = ${eps} from ${log_file}" 
    # final_acc=$(awk -F ':' '/^FINAL_ROBUST_ACC:/ {print $2}' "${log_file}")
    # ... (removed result extraction and CSV logging) ...
    
    echo "Finished experiment for epsilon = ${eps}. Results logged to MLflow."
done

echo "-------------------------------------------------"
echo "Epsilon sweep finished."
echo "Check MLflow UI for detailed results."
echo "Individual logs saved in: ${LOG_DIR}" 
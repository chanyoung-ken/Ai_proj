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

# Output CSV file path
CSV_FILE="epsilon_robustness_results.csv"

# Directory for individual run logs
LOG_DIR="epsilon_run_logs"
mkdir -p "${LOG_DIR}"

# Number of epochs for each run
EPOCHS=100 # Or you can set this per epsilon if needed

# --- CSV Header ---
echo "epsilon,best_robust_acc" > "${CSV_FILE}"

# --- Run Experiments ---
echo "Starting epsilon sweep... Results will be saved to ${CSV_FILE}"

for eps in "${EPSILON_VALUES[@]}"; do
    echo "-------------------------------------------------"
    echo "Running experiment for epsilon = ${eps}"
    echo "-------------------------------------------------"
    
    log_file="${LOG_DIR}/run_eps_${eps//./_}.log"
    
    # Run the python script, redirect stdout and stderr to log file
    # Use tee to also see output on console
    python "${PYTHON_SCRIPT}" --adv_eps "${eps}" --epochs "${EPOCHS}" 2>&1 | tee "${log_file}"
    
    # Check the exit code of the python script (from the left side of the pipe)
    script_exit_code=${PIPESTATUS[0]}
    if [ ${script_exit_code} -ne 0 ]; then
        echo "Error: Python script for epsilon ${eps} failed with exit code ${script_exit_code}. Check log file: ${log_file}" >&2
        # Optionally continue to the next epsilon or exit the script
        # exit 1 # Uncomment to stop the script on error
        continue # Continue with the next epsilon
    fi
    
    echo "Extracting results for epsilon = ${eps} from ${log_file}"
    
    # Extract the final robust accuracy using grep and cut/awk
    # Use grep -oP for Perl-compatible regex with positive lookbehind
    # Or use simpler grep | cut
    # final_acc=$(grep "FINAL_ROBUST_ACC:" "${log_file}" | cut -d ':' -f 2)
    
    # Alternative using awk (often more robust)
    final_acc=$(awk -F ':' '/^FINAL_ROBUST_ACC:/ {print $2}' "${log_file}")
    
    if [ -z "${final_acc}" ]; then
        echo "Warning: Could not extract final accuracy for epsilon ${eps} from log file: ${log_file}" >&2
    else
        echo "Epsilon ${eps} | Best Robust Acc: ${final_acc}"
        # Append result to CSV
        echo "${eps},${final_acc}" >> "${CSV_FILE}"
    fi
    
    echo "Finished experiment for epsilon = ${eps}"
done

echo "-------------------------------------------------"
echo "Epsilon sweep finished."
echo "Results saved in: ${CSV_FILE}"
echo "Individual logs saved in: ${LOG_DIR}" 
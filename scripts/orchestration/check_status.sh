#!/bin/bash
################################################################################
# Windstorm Institute Paper 7 - Status Checker
# Quickly check progress of all experiments
################################################################################

BASEDIR="/home/user1-gpu/agi-extensions"

echo "================================================================================"
echo "WINDSTORM INSTITUTE PAPER 7 - EXPERIMENT STATUS"
echo "================================================================================"
echo "Checked: $(date)"
echo ""

# Function to check experiment status
check_experiment() {
    local exp_num=$1
    local exp_name=$2
    local log_file="$BASEDIR/exp-${exp_num}/exp${exp_num}.log"
    local results_dir="$BASEDIR/exp-${exp_num}/results"

    echo "Experiment $exp_num: $exp_name"
    echo "----------------------------------------"

    # Check if running
    if pgrep -f "exp${exp_num}_" > /dev/null; then
        echo "Status: RUNNING ✓"
        echo "Process: $(pgrep -f "exp${exp_num}_")"

        # Show last log line
        if [ -f "$log_file" ]; then
            echo "Latest: $(tail -1 $log_file)"
        fi
    else
        # Check if completed
        if [ -d "$results_dir" ] && [ "$(ls -A $results_dir 2>/dev/null)" ]; then
            echo "Status: COMPLETED ✓"
            echo "Results: $(ls -1 $results_dir | wc -l) files"
            ls -1 "$results_dir" | head -5
            [ "$(ls -1 $results_dir | wc -l)" -gt 5 ] && echo "..."
        else
            echo "Status: NOT STARTED"
        fi
    fi

    echo ""
}

# Check all experiments
check_experiment 3 "Recurrent vs Transformer"
check_experiment 2 "Quantization Cliff"
check_experiment 6 "Energy Survey"
check_experiment 1 "Synthetic Training"

echo "================================================================================"
echo "GPU STATUS"
echo "================================================================================"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv

echo ""
echo "================================================================================"
echo "DISK USAGE"
echo "================================================================================"
du -sh $BASEDIR/exp-*/
du -sh $BASEDIR

echo ""
echo "================================================================================"

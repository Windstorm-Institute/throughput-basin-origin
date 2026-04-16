#!/bin/bash
################################################################################
# Windstorm Institute Paper 7: AGI Extensions of the Throughput Basin Framework
# Master Experiment Orchestration Script
################################################################################

set -e  # Exit on error

BASEDIR="/home/user1-gpu/agi-extensions"
LOGDIR="$BASEDIR/logs"

mkdir -p "$LOGDIR"

echo "================================================================================"
echo "WINDSTORM INSTITUTE PAPER 7 - EXPERIMENT SUITE"
echo "AGI Extensions of the Throughput Basin Framework"
echo "================================================================================"
echo ""
echo "Principal Investigator: Grant Lavell Whitmer III"
echo "Platform: Veron 1 (RTX 5090, 32GB VRAM)"
echo "Started: $(date)"
echo ""
echo "This script will run all four experiments in sequence:"
echo "  - Experiment 3: Recurrent vs Transformer (4-8 hours)"
echo "  - Experiment 2: Quantization Cliff (6-10 hours)"
echo "  - Experiment 6: Energy Survey (4-6 hours)"
echo "  - Experiment 1: Synthetic Training (8-12 hours)"
echo ""
echo "Total estimated runtime: 22-36 hours"
echo "================================================================================"
echo ""

# Function to run experiment with logging
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local script=$3

    echo ""
    echo "###############################################################################"
    echo "# EXPERIMENT $exp_num: $exp_name"
    echo "###############################################################################"
    echo "Started: $(date)"
    echo ""

    python3 "$script" 2>&1 | tee "$LOGDIR/exp${exp_num}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "Completed: $(date)"
    echo "-------------------------------------------------------------------------------"
}

################################################################################
# EXPERIMENT 3: RECURRENT VS TRANSFORMER
################################################################################

echo "Checking if Experiment 3 is already running..."
if pgrep -f "exp3_main.py" > /dev/null; then
    echo "Experiment 3 is already running. Waiting for completion..."
    while pgrep -f "exp3_main.py" > /dev/null; do
        sleep 60
        echo "Still running... $(date)"
    done
    echo "Experiment 3 completed."
else
    echo "Starting Experiment 3..."
    run_experiment 3 "Recurrent vs Transformer" "$BASEDIR/exp-3/code/exp3_main.py"
fi

# Generate Experiment 3 summary
echo "Generating Experiment 3 summary..."
python3 "$BASEDIR/exp-3/code/generate_summary.py"

################################################################################
# EXPERIMENT 2: QUANTIZATION CLIFF
################################################################################

run_experiment 2 "Quantization Cliff" "$BASEDIR/exp-2/code/exp2_main.py"

################################################################################
# EXPERIMENT 6: ENERGY SURVEY / THERMODYNAMIC ROADMAP
################################################################################

run_experiment 6 "Energy Survey / Thermodynamic Roadmap" "$BASEDIR/exp-6/code/exp6_main.py"

################################################################################
# EXPERIMENT 1: SYNTHETIC TRAINING BASELINE
################################################################################

echo ""
echo "###############################################################################"
echo "# EXPERIMENT 1: SYNTHETIC TRAINING BASELINE"
echo "###############################################################################"
echo ""

echo "Phase 1: Generating Synthetic Corpora..."
python3 "$BASEDIR/exp-1/code/exp1_generate_corpora.py" 2>&1 | tee "$LOGDIR/exp1_corpora.log"

echo ""
echo "Phase 2-3: Training Tokenizers and Models..."
python3 "$BASEDIR/exp-1/code/exp1_train.py" 2>&1 | tee "$LOGDIR/exp1_train.log"

echo ""
echo "Phase 4: Evaluation..."
python3 "$BASEDIR/exp-1/code/exp1_evaluate.py" 2>&1 | tee "$LOGDIR/exp1_eval.log"

################################################################################
# FINAL SYNTHESIS
################################################################################

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================================"
echo "Completed: $(date)"
echo ""
echo "Results are available in:"
echo "  - $BASEDIR/exp-1/results/"
echo "  - $BASEDIR/exp-2/results/"
echo "  - $BASEDIR/exp-3/results/"
echo "  - $BASEDIR/exp-6/results/"
echo ""
echo "Next steps:"
echo "  1. Review individual experiment summaries"
echo "  2. Run synthesis script to generate Paper 7 master summary"
echo "  3. Initialize git repository and push to GitHub"
echo ""
echo "================================================================================"

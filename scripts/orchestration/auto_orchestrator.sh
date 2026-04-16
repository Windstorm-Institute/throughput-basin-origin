#!/bin/bash
################################################################################
# Windstorm Institute Paper 7 - Autonomous Orchestration
# Monitors running experiments and auto-launches next phases
################################################################################

BASEDIR="/home/user1-gpu/agi-extensions"
LOGDIR="$BASEDIR/logs"
mkdir -p "$LOGDIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGDIR/orchestrator.log"
}

log "=============================================================================="
log "AUTONOMOUS ORCHESTRATOR STARTED"
log "=============================================================================="

# Track completion states
EXP1_PHASE1_DONE=false
EXP1_PHASE23_LAUNCHED=false
EXP1_PHASE23_DONE=false
EXP1_PHASE4_LAUNCHED=false
EXP2_DONE=false
EXP6_LAUNCHED=false

while true; do

    # Check Experiment 1 Phase 1 (Corpus Generation)
    if [ "$EXP1_PHASE1_DONE" = false ]; then
        if ! pgrep -f "exp1_generate_corpora.py" > /dev/null; then
            # Check if corpus files exist
            if [ -f "$BASEDIR/exp-1/corpora/syn2.txt" ] &&
               [ -f "$BASEDIR/exp-1/corpora/syn4.txt" ] &&
               [ -f "$BASEDIR/exp-1/corpora/syn8.txt" ]; then
                log "✓ Experiment 1 Phase 1 COMPLETE - Corpora generated"
                EXP1_PHASE1_DONE=true

                # Launch Phase 2-3 (Training)
                if [ "$EXP1_PHASE23_LAUNCHED" = false ]; then
                    log "→ Launching Experiment 1 Phase 2-3 (Training)..."
                    cd "$BASEDIR/exp-1/code"
                    nohup python3 exp1_train.py > "$BASEDIR/exp-1/exp1_train.log" 2>&1 &
                    log "  Started training (PID: $!)"
                    EXP1_PHASE23_LAUNCHED=true
                fi
            fi
        fi
    fi

    # Check Experiment 1 Phase 2-3 (Training)
    if [ "$EXP1_PHASE23_LAUNCHED" = true ] && [ "$EXP1_PHASE23_DONE" = false ]; then
        if ! pgrep -f "exp1_train.py" > /dev/null; then
            # Check if models were trained
            if [ -d "$BASEDIR/exp-1/models/syn2/final" ] &&
               [ -d "$BASEDIR/exp-1/models/syn8/final" ]; then
                log "✓ Experiment 1 Phase 2-3 COMPLETE - Models trained"
                EXP1_PHASE23_DONE=true

                # Launch Phase 4 (Evaluation)
                if [ "$EXP1_PHASE4_LAUNCHED" = false ]; then
                    log "→ Launching Experiment 1 Phase 4 (Evaluation)..."
                    cd "$BASEDIR/exp-1/code"
                    nohup python3 exp1_evaluate.py > "$BASEDIR/exp-1/exp1_eval.log" 2>&1 &
                    log "  Started evaluation (PID: $!)"
                    EXP1_PHASE4_LAUNCHED=true
                fi
            fi
        fi
    fi

    # Check Experiment 2 (Quantization)
    if [ "$EXP2_DONE" = false ]; then
        if ! pgrep -f "exp2_main.py" > /dev/null; then
            # Check if results exist
            if [ -f "$BASEDIR/exp-2/results/exp2_quantization.csv" ]; then
                log "✓ Experiment 2 COMPLETE - Quantization cliff identified"
                EXP2_DONE=true

                # Launch Experiment 6 (Energy Survey)
                if [ "$EXP6_LAUNCHED" = false ]; then
                    log "→ Launching Experiment 6 (Energy Survey)..."
                    cd "$BASEDIR/exp-6/code"
                    nohup python3 exp6_main.py > "$BASEDIR/exp-6/exp6_run.log" 2>&1 &
                    log "  Started energy survey (PID: $!)"
                    EXP6_LAUNCHED=true
                fi
            fi
        fi
    fi

    # Check if all experiments are complete
    if [ "$EXP1_PHASE4_LAUNCHED" = true ] &&
       ! pgrep -f "exp1_evaluate.py" > /dev/null &&
       [ "$EXP6_LAUNCHED" = true ] &&
       ! pgrep -f "exp6_main.py" > /dev/null; then

        # Verify all results exist
        if [ -f "$BASEDIR/exp-1/results/exp1_self_eval.csv" ] &&
           [ -f "$BASEDIR/exp-2/results/exp2_quantization.csv" ] &&
           [ -f "$BASEDIR/exp-3/results/exp3_bpt_comparison.csv" ] &&
           [ -f "$BASEDIR/exp-6/results/exp6_energy.csv" ]; then

            log "=============================================================================="
            log "ALL EXPERIMENTS COMPLETE!"
            log "=============================================================================="
            log ""
            log "Results available in:"
            log "  - $BASEDIR/exp-1/results/"
            log "  - $BASEDIR/exp-2/results/"
            log "  - $BASEDIR/exp-3/results/"
            log "  - $BASEDIR/exp-6/results/"
            log ""
            log "Next step: Run synthesis"
            log "  python3 $BASEDIR/synthesize_results.py"
            log ""

            # Create completion marker
            touch "$BASEDIR/ALL_EXPERIMENTS_COMPLETE.flag"

            exit 0
        fi
    fi

    # Status update every 5 minutes
    sleep 300

    log "Status check..."
    log "  Exp 1 Phase 1 (Corpora): $([ "$EXP1_PHASE1_DONE" = true ] && echo "DONE" || echo "RUNNING")"
    log "  Exp 1 Phase 2-3 (Training): $([ "$EXP1_PHASE23_DONE" = true ] && echo "DONE" || ([ "$EXP1_PHASE23_LAUNCHED" = true ] && echo "RUNNING" || echo "WAITING"))"
    log "  Exp 1 Phase 4 (Eval): $([ "$EXP1_PHASE4_LAUNCHED" = true ] && echo "RUNNING" || echo "WAITING")"
    log "  Exp 2 (Quantization): $([ "$EXP2_DONE" = true ] && echo "DONE" || echo "RUNNING")"
    log "  Exp 6 (Energy): $([ "$EXP6_LAUNCHED" = true ] && echo "RUNNING" || echo "WAITING")"

done

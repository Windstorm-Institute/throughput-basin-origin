#!/bin/bash
# Wait for Phase 1-2 orchestrator to finish
ORCH_PID=$(cat /home/user1-gpu/agi-extensions/weekend_experiments/orchestrator_pid.txt 2>/dev/null)
if [ -n "$ORCH_PID" ]; then
    echo "Waiting for Phase 1-2 orchestrator (PID $ORCH_PID) to finish..."
    while kill -0 $ORCH_PID 2>/dev/null; do
        sleep 60
    done
    echo "Phase 1-2 complete. Starting Phase 3-5..."
fi

cd /home/user1-gpu/agi-extensions
python3 -u weekend_experiments/phase3_audio_and_controlled_entropy.py

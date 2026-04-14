#!/bin/bash
# Wait for final round to finish
PID=$(cat /home/user1-gpu/agi-extensions/weekend_experiments/final_round_pid.txt 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Waiting for final round (PID $PID)..."
    while kill -0 $PID 2>/dev/null; do sleep 120; done
fi
echo "Final round done. Remaining experiments (5-9) need separate scripts."
echo "$(date): Final round chain activated." >> /home/user1-gpu/agi-extensions/weekend_experiments/final_round/orchestrator.log

#!/bin/bash
# Wait for Round 3 to finish, then run remaining experiments
R3_PID=$(cat /home/user1-gpu/agi-extensions/weekend_experiments/round3_pid.txt 2>/dev/null)
if [ -n "$R3_PID" ]; then
    echo "Waiting for Round 3 (PID $R3_PID)..."
    while kill -0 $R3_PID 2>/dev/null; do sleep 120; done
    echo "Round 3 complete. Starting overnight experiments..."
fi

# Exp 5, 6, 8 are GPU-heavy overnight runs
# For now, just log that they're queued — the scripts will be written
echo "$(date): Round 3 chain activated. Remaining experiments (5,6,8,9) need manual launch or script creation." >> /home/user1-gpu/agi-extensions/weekend_experiments/round3.log

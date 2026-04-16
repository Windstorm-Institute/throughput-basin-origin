#!/usr/bin/env python3
"""Run just GS1 (Visual MAE from scratch) with the seed fix."""
import sys
sys.path.insert(0, '/home/user1-gpu/agi-extensions')
from weekend_experiments.grandslam import gs1_visual_mae, log, gpu_log

log("="*70)
log("GS1 RELAUNCH — Visual MAE from scratch (seed fix applied)")
log("="*70)
gpu_log()
gs1_visual_mae()
log("GS1 RELAUNCH COMPLETE")

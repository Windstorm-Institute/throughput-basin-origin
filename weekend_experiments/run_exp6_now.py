#!/usr/bin/env python3
"""Run Exp 6 (Lloyd-Max INT3 end-to-end) immediately — GPU has enough VRAM."""
import sys
sys.path.insert(0, '/home/user1-gpu/agi-extensions')
from weekend_experiments.decisive_phase2 import exp6_lloydmax_int3, log

log("Running Exp 6 directly (10.8 GB free, sufficient for Pythia-410M)")
exp6_lloydmax_int3()
log("DONE")

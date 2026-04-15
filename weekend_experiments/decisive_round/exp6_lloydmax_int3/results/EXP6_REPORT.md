# Exp 6: Lloyd-Max INT3 End-to-End BPT

| Method | Bits | BPT | Operational? |
|---|---|---|---|
| FP16 | 16 | 4.2688 | ✅ |
| lloyd_max | 4 | 8.5084 | ❌ |
| lloyd_max | 3 | 11.7377 | ❌ |
| symmetric | 4 | 16.8923 | ❌ |
| symmetric | 3 | 16.0500 | ❌ |

## Verdict
**Lloyd-Max INT3 FAILS end-to-end (BPT=11.74).** Per-matrix cosine 0.965 doesn't survive error accumulation.

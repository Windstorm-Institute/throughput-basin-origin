# Exp 2: Quantization Method Comparison

## Key finding: The cliff location depends on the method

| Method | Precision | BPT | vs FP16 |
|---|---|---|---|
| FP16 | 16 | 4.2688 | 1.00× |
| symmetric | 8 | 4.8248 | 1.13× |
| symmetric | 4 | 16.8923 | 3.96× |
| symmetric | 3 | 16.0500 | 3.76× |

## Interpretation

- bitsandbytes NF4 uses non-uniform quantization levels optimized for normal distributions
- Symmetric quantization uses uniform levels — cruder but matches real hardware
- If NF4 works at INT4 but symmetric doesn't → the cliff is about LEVEL ALLOCATION, not just bit count
- This refines Paper 9's claim: the floor is 4 bits for uniform quantization; NF4-style methods can push lower

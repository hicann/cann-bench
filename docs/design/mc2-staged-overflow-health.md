# MC2 staged-reference overflow health (2026-09-29)

## Reproduction

AllGatherMatmul final report `cann_final_eval_20260929_194757` has 18 passes,
no candidate failures, and reference-invalid cases 25/35. Both ranks fail with
`CPU staged cascade contains nonfinite values`; the gathered auxiliary outputs
pass with zero error. Candidate branch revision is `56c63c81a`, benchmark
revision before this fix is `9f3c4d3`.

These FP16 cases transpose the weight, using K=12288/8192 and seeds 5/15.
Rank-0 input and weight use the same random seed, so corresponding first rows
produce a sum of squares. The first FP32 dot products are approximately
102078.70 and 67597.66. Both overflow FP16. The small bias in case35 cannot
bring its first dot product back into the representable range. Both ranks
receive rank-0 data through AllGather.

The mathematical reference correctly remains FP32; the independent CPU
cascade correctly models intermediate FP16 rounding. The bug was rejecting
every nonfinite staged value before applying the existing signed-overflow
rules in `mc2_compare.error_metrics`.

## Repair and invariants

- Mathematical FP32 G must still be finite.
- CPU-stage and NPU exceptional-value positions and signs must each match
  the existing output-dtype range contract. NaN, unexpected/missing Inf,
  wrong signs or positions remain failures.
- For reference health, accumulate G squared, stage-minus-G squared, and
  NPU-minus-stage squared on the same finite intersection, normalizing by
  its size. This avoids Inf-minus-Inf and prevents large or numerous overflow
  positions from diluting errors in remaining finite values.
- The all-finite path retains its previous stage-error statistic. All-overflow
  outputs have zero finite error but still require exact exceptional masks.
- Candidate metrics, tolerance constants, task inputs, kernel source, scoring
  formula, and the generic non-staged reference policy are unchanged.
- Report `finite_health_count` and `expected_overflow_count` in reference health.

## Regression coverage

`tests/ut/test_mc2_staged_overflow.py` covers FP16/BF16 positive and negative
overflow, all-overflow outputs, exceptional masks on candidate/NPU/staged sides,
finite candidate errors, malformed mathematical references, FP32 comparison
override, intermediate overflow with finite final math, and errors hidden among
4096 expected overflow values (both relative and near-zero absolute cases).

`tests/ut/test_mc2_allgather_staged_reference.py` also reproduces cases25/35 for
both ranks with the original K, seed, transpose, bias, dtype and ranges, reducing
M/N only to keep CPU regression inexpensive. It invokes separate low-precision
CPU matmul/add for the emulated native output, not simply `native = staged`.

Final local MC2 regression: **637 passed in 32.96s**; `git diff --check` clean.
Local verification is CPU-only. The NPU native and candidate still need a new
20-case evaluation. Do not edit previous reports to mark blocked cases passed.

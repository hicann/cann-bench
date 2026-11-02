# Optional tensor fixtures for kernel_eval

The cache replaces deterministic input generation and trusted reference execution
for explicitly selected tasks. Candidate execution, output dtype contracts,
accuracy thresholds, profiling and fresh-input retries keep their current behavior.
The default is `off`. No tensor fixtures are committed to this repository.

## Format decision

Use Hugging Face **safetensors**, through its PyTorch API, for tensor bytes and a
versioned JSON manifest for identity and structure. Safetensors is not a PyTorch
checkpoint format and does not deserialize Python objects. It supports the real,
integer and boolean types needed here, named tensors, scalars and empty tensors.
Its built-in metadata is string-to-string, so the structured contract belongs in
the manifest. No pickle, ZIP decompression or arbitrary Python object hooks are
needed on the evaluation read path.

One case/seed has one immutable directory:

```text
<cache-root>/<sha256-of-canonical-identity>/
    manifest.json
    inputs.safetensors
    references.safetensors
```

Separate input and reference files make the roles explicit and avoid a monolithic
bank that encourages retaining all cases. The initial reader loads one selected
case at a time. Existing profiler queues can still retain inputs: enabling a
cache does not remove the need for bounded batch sizes and memory admission.

The manifest has `format="cann-bench.tensor-cache"`, `version=1`, `key`,
`identity`, `producer`, and `files`. Each file has its byte length, SHA256,
physical tensor-name → `{shape, dtype}` map, a `tensor_objects` map from logical
object names to physical tensor names, and a value tree. Trees refer to logical
objects and use exactly one of:

```json
{"tensor": "object_000000"}
{"scalar": null}
{"list": [{"tensor": "object_000000"}, {"scalar": null}]}
{"tuple": [{"tensor": "object_000000"}, {"tensor": "object_000001"}]}
```

Scalar leaves may be null, bool, int, finite float or string. Tensor NaNs/Infs
retain their exact bits. Lists, tuples, optional outputs, rank-zero tensors and
empty tensors preserve their structure. Arbitrary objects/dictionaries are not
part of v1. The input tree is the ordered input list. The reference tree is a
two-item tuple `(oracle_outputs, native_outputs)` **before** `get_output`.
The evaluator applies `get_output` to candidate and both references as usual.
No candidate outputs are written to the cache.

For example, `tensor_objects={"object_000000": "tensor_000000",
"object_000001": "tensor_000000"}` stores equal independent tensors once on disk,
but restores two independent writable tensors. Repeated uses of the same logical
object name preserve actual Tensor object aliases, including `native_cpu` reusing
the oracle. Missing mappings, unknown physical names and unreferenced logical or
physical tensors are invalid. Earlier experimental v1 banks without this mapping
must be rebuilt; changed harness source fingerprints prevent their reuse.

The safetensors header is also checked against the manifest. Unknown versions,
bad hashes, mismatched names/dtypes/shapes, missing roles or malformed trees fail
the case, even in `prefer` mode. Only a genuinely absent identity can fall back.
Builders write/fsync complete files in a temporary sibling directory, then rename
the directory atomically. Existing entries are validated and never overwritten.
Concurrent builders may compute redundantly, but readers never see a partial
published fixture. Only temporary directories created by the current builder are
removed on failure.

## Dtypes and why there are two reference roles

| Role | Storage policy | LCE example |
| --- | --- | --- |
| Inputs | Preserve actual case dtype; no FP64 copies or intermediates | BF16 hidden/weight, INT32 labels |
| Oracle | Preserve the actual oracle result dtype | FP64 loss |
| Native reference | Preserve the result of the actual native/bench computation | FP32 loss, even for BF16 inputs |
| Candidate | Never cached; execute each evaluation | Sample output |

V1 supports BOOL, U8/U16/U32/U64, I8/I16/I32/I64, F16, BF16, F32 and F64.
This matches the current data generator's dtype mapping. Complex, packed,
quantized, sparse and device-specific tensor representations are excluded.
There is no implicit dtype conversion during serialization or loading.

The distinction is **reference role**, not simply high/low storage precision:

* The default relative-error checker consumes the oracle and the native reference
  to calibrate small-value, cancellation and related error-count fallbacks.
* `oracle.to(target_dtype)` models final rounding only. It cannot reproduce the
  rounding and accumulation error of an actual native computation. Substituting
  that cast can change a candidate's pass/fail result.
* A BF16 input does not imply a BF16 output. LCE's native function returns FP32.
  Output dtype contracts are still checked against the candidate's actual output.
* `native_cpu` already reuses its golden as native reference. The two roles can
  point to one tensor. The writer also deduplicates byte-identical reference leaves
  with the same dtype and shape, including integer/bool outputs. Physical dedup
  does not merge independent logical objects: an in-place `get_output` must see
  the same Tensor aliases as the uncached references. Distinct reference Tensor
  objects sharing one backing storage (including disjoint views) are unsupported
  and rejected by the builder; v1 does not encode view offsets or storage aliases.
* An oracle-only checker such as `allclose` does not need the second role to decide
  accuracy. V1 nevertheless retains both logical roles to allow the existing
  evaluator/checker selection without changing semantics. An oracle-only storage
  policy can be a separate schema/policy extension; it must not silently switch
  the current checker to a different acceptance rule.

For LCE, the two outputs together are only **12 bytes per case**, or **240 bytes
for all 20 cases**. The input bank dominates: about **28.023 GiB / 30.09 GB** for
one seed, plus small headers/manifests. Minimum case 18 is about **258 MiB**;
maximum-compute case 16 is **2.383 GiB**. FP32 case 11 has the largest input
storage, about **3.479 GiB**. Saving only the FP64 scalar would not materially
reduce this bank. These are uncompressed payload estimates from the LCE study,
not claimed timings for this safetensors implementation.

## Identity, portability and trust

Identity includes task/case/operator, ordered shapes and dtypes, value ranges,
attrs, the resolved deterministic seed, normalized input distribution, golden
precision strategy, PyTorch version, task source/contract hashes and all
`kernel_eval` Python source hashes. Paths are relative, not workstation-specific.
Performance baselines, candidate code and device timing are not cache identity.
Source hashing works on an installed source bundle without a Git database.
Identity JSON represents non-finite contract floats with typed values:
`{"__cann_bench_nonfinite__": ["float", "nan"]}` (or `"inf"` / `"-inf"`).
Literal contract dictionaries containing that reserved key are recursively
escaped as `{"__cann_bench_nonfinite__": ["dict", <encoded dictionary>]}`,
so strings or dictionaries that resemble these markers cannot share an identity
with a special float. Ordinary finite JSON values keep their representation;
tuples normalize to lists. The same normalized identity is hashed, written and
compared on load; manifest writing still rejects non-standard JSON numeric literals.
Hashing the whole harness is conservative: unrelated harness edits invalidate
fixtures too. A future narrower dependency manifest can improve reuse.
Source digests use a bounded process-local memo keyed by path, device/inode,
size, mtime and ctime. Each identity request still rescans the source file list
and stats each file, detecting additions, removals, replacements and edits
(including same-size edits with a restored mtime). Unchanged source contents
are not reread for every case. Source changes during hashing are rejected.
This optimization does not memoize tensor payload checksums: those are verified
on every load. Source trees are trusted and should remain fixed during a run.

Reference producer thread counts and the PyTorch/BLAS build configuration are
recorded as provenance, not compared with the reader host. The fixture defines
the reference values produced by that environment; it does not promise a
bit-identical re-computation on every CPU/BLAS/thread configuration. A PyTorch
version change currently requires rebuilding. Different eval seeds produce
different banks. `eval_seed=None` is rejected for cache-enabled evaluation.

The reader checks file hashes on every load and copies tensor data into owned,
writable CPU RAM before use; it does not retain mmap-backed candidate inputs.
Changing an input in one evaluation cannot modify the persisted fixture.
The copy and integrity scan are real costs and belong in any performance
measurement; timing only mmap construction would be misleading.

Fixtures are trusted benchmark assets. Safetensors avoids Python-object loading;
a checksum is integrity metadata, not authentication of an attacker-controlled
manifest. Build outside submission execution, distribute with trusted manifests,
and expose the cache read-only to evaluation. This feature is not a process
isolation boundary or a guarantee that fixed answers cannot be memorized.
Existing fresh-input accuracy retry deliberately bypasses cached inputs and
references; retain fresh checks and private seed banks as appropriate.

## Initial scope

V1 supports selected deterministic CANN tasks with CPU references and contiguous,
non-aliased dense inputs (including optional `None` slots). Cache building and
`require` reject `get_input` hooks, distributed/MC2 tasks, non-CANN benches and
native-NPU reference policies. In `prefer`, these unsupported cases log a warning
and follow their existing uncached evaluation path, including MC2 dispatch.
Reference functions must not mutate inputs; the builder checks input hashes before
and after reference execution and rejects mutations. Keep cache files outside the
selected task source directory so they cannot become part of their own identity.
Silently making a strided/aliased tensor contiguous could change what a kernel is
being tested on, so those cases need a future layout-aware format extension.
Tasks outside the explicit selection follow the original generation path.

## Build and use

Install the optional dependency:

```bash
uv sync --extra tensor-cache
# Or in an existing environment:
pip install 'safetensors>=0.6.2,<1'
```

Build only from trusted tasks; the command does not load or execute a submission:

```bash
PYTHONPATH=src python -m kernel_eval.cache_cli build \
  --task level4/linear_cross_entropy --case-id 18 \
  --eval-seed 0 --threads 8 --cache-dir /workspace/cann-bench-cache
```

Omit `--case-id` to build all cases in that task. Repeat `--task` to select more
tasks. `--tasks-root` selects a custom task root. Each case publishes separately;
an existing valid fixture is reused. A failed build exits nonzero.

Add these flags to the usual `kernel_eval.cli eval` or `kernel_eval.staged_eval`
invocation (including its existing source/task/device arguments):

```bash
--tensor-cache-dir /workspace/cann-bench-cache \
--tensor-cache-mode prefer \
--tensor-cache-task level4/linear_cross_entropy
```

`prefer` reads a matching fixture and computes normally on a miss or unsupported
case; `require` fails selected cases on a miss or unsupported case; `off` is the
default. Corrupt fixtures, invalid configuration and source fingerprint errors
fail in both enabled modes; they are not treated as unsupported cases.
Evaluation never populates the cache. Options are forwarded to evaluation child
processes and both staged evaluation phases. Console messages identify hits,
keys, misses and unsupported-case fallbacks.

## Validation

Tests cover exact dtype/shape/tree round trips, integer/float/reference roles,
corrupt/stale/partial fixtures, task and seed selection, mutable input isolation,
atomic publication, cached/uncached evaluator results, native-reference error
calibration, fresh retry and CLI/worker/staged propagation. Unsupported layouts
must fail clearly. Large NPU timings require a device follow-up; unit tests use
small real CPU tensors and references.

References: [safetensors format](https://github.com/huggingface/safetensors#format),
[PyTorch API](https://huggingface.co/docs/safetensors/api/torch),
[shared tensor limitations](https://huggingface.co/docs/safetensors/torch_shared_tensors).

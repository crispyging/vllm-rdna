# PR #18: GPTQ tile-size experiment

## Decision

Do not merge the global 128-to-256 constant change as written. A narrow
small-M optimization has merit, but requires architecture and shape gating,
retaining the existing tile elsewhere, and a serving A/B on an affected path.
No production source, installed extension, launcher, or PR #18 was changed.

## Method

Physical GPU 8 on par1-cs25, Radeon PRO V620 gfx1030, ROCm 7.14, PyTorch
2.12.0+rocm7.14.0. Other allocated GPUs were occupied; they were not used or
modified. The whole machine was not exclusively reserved.

The current hipified `q_gemm` source was compiled twice with the existing
CMake compiler flags, changing BLOCK_KN_SIZE to 128 or 256. Separate namespaces,
entry symbols and operator registrations allow both libraries in one process.
Headers and math are otherwise shared. Exact commands, library hashes, source
hashes and archived scripts accompany this receipt. Scripts contain this
experiment's absolute paths and are evidence, not portable serving tools.

Five K/N shapes: 5120/8192, 5120/5120, 17408/5120, 2560/7168, 3072/2560.
Primary matrix: M=1/3/4/8/16/24/32/64/256, group sizes 32/128, GPTQv1/v2,
4-bit weights, use_exllama=True, eager and captured replay. Six alternating
A/B rounds, 100 calls per timing sample, CUDA/HIP events, median reported.
Large-prefill matrix: M=512/4096, group128, GPTQv1, the same five shapes and
both execution modes; four alternating rounds, 20 calls per sample.
Synthetic weights and inputs, fixed seed 1801. No model-level quality or
serving throughput result is claimed. Times include the op's output zeroing
and dispatch; graph replay removes repeated host operator submission.

Correctness compares both operators with an FP32 matmul reference of the
same dequantized weights. The two shuffle outputs must agree exactly.
Primary-matrix acceptance: relative L2 below 0.05 versus reference, below 0.02
between variants. Maximum observed reference error was 0.018753; maximum
between-variant relative L2 was 0.002210. This is tolerance parity, not bitwise
identity. All 360 primary and 20 large-prefill timing cells completed.

## Findings

For GPTQv2/group128 graph replay at M=1, the three shapes in the PR measured:

- K5120/N8192: 49.073 to 33.004 microseconds, 1.487x throughput.
- K5120/N5120: 40.206 to 30.440 microseconds, 1.321x throughput.
- K17408/N5120: 93.475 to 72.957 microseconds, 1.281x throughput.

Across all 20 shape/group/format combinations in graph mode, the geometric
mean speed ratio is 1.202 at M=1, 1.139 at M=3, 1.102 at M=4, but 0.956 at
M=8, 0.970 at M=16, 0.962 at M=24, and 0.981 at M=32. These are unweighted
kernel-case summaries, not a workload-weighted serving estimate.

The win is not universal even at M=1: the lowest graph ratio there was 0.931.
For the Flash-Next K2560/N7168 shape, GPTQv2/group128 graph mode at M=8 went
from 41.496 to 46.395 microseconds (11.8% longer). K3072/N2560 at M=8 went
from 39.372 to 44.052 microseconds (11.9% longer).

At M=512 and 4096, every tested graph-mode speed difference was below 1%
in magnitude, with no demonstrated meaningful prefill gain.

## Relevance and limits

The RDNA2 linear dispatcher prefers specialized RDNA2 kernels for small M.
Its Exllama fallback is used in selected cases and large GPTQ prefill; this
experiment cannot turn a direct M=1 Exllama gain into a default Flash-Next
serving claim. The deterministic offload launcher further redirects its
Exllama choice through the deterministic RDNA path.

The original PR changes an unconditional macro in a source also built for
CUDA. It does not implement the RDNA-only scope claimed in its description.
This experiment used only gfx1030, 4-bit, use_exllama=True. CUDA, other AMD
architectures, 2/3/8-bit, use_exllama=False, and arbitrary tail shapes remain
unqualified. No whole-model A/B or measured gain beyond isolated operator
execution is established. No clocks/power isolation or independent whole-run
repeat was performed; small differences should be treated as inconclusive.

## Receipts

`results.json` contains all primary samples, errors and environment metadata.
`prefill-results.json` contains the large-prefill samples. `compile*.json`
records exact build commands; `source-manifest.txt` records native-source
hashes. The `.txt` script archives reproduce the experiment on its original
isolated remote layout. Runtime code and PR #18 remain unchanged.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end check of an rdna_hint_flex checkpoint in vLLM against its dense export.

Run once on the rdna_hint_flex checkpoint and once on ``hi2 export-dense`` of it (same
weights, dequantized to fp16), then ``--compare`` the two JSON files:

* teacher-forced prompt log-probabilities on a fixed text (mean NLL and the
  per-token gap): the rdna_hint_flex kernels should reproduce the dense model almost
  exactly, since only the fp16 rounding of the transformed activation differs;
* greedy continuations of a few prompts (coherence, first divergence).

  python tools/rdna/rdna_hint_flex/e2e_check.py --model <dir> --out a.json [--eager]
  python tools/rdna/rdna_hint_flex/e2e_check.py --compare a.json b.json
"""

import argparse
import json
import math
import time

PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the factorial of n.\n\n```python\n",
    "Q: What is 17 + 25?\nA:",
    "Explain in two sentences why the sky is blue.",
]

TEXT = (
    "A graphics processing unit is a specialized electronic circuit designed "
    "to accelerate the processing of images and video. Modern GPUs are also "
    "used for general purpose computation, because their highly parallel "
    "structure makes them efficient for algorithms that process large blocks "
    "of data in parallel, such as the matrix multiplications at the heart of "
    "neural networks. Memory bandwidth often limits how fast a language model "
    "can generate text, which is why weight quantization, storing each weight "
    "in a few bits instead of sixteen, speeds up decoding on consumer cards. "
    "The trade-off is accuracy: fewer bits mean a coarser approximation of "
    "the original weights, and careful calibration is needed to keep the "
    "model's answers close to those of the full-precision network."
)


def run(args) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=args.eager,
        seed=0,
        max_num_seqs=args.max_num_seqs,
        language_model_only=args.language_model_only,
    )
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, greedy)
    gen_s = time.perf_counter() - t0
    tf = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    pl = llm.generate([TEXT], tf)[0]
    ids = pl.prompt_token_ids
    lps = [
        None if d is None else d[ids[i]].logprob
        for i, d in enumerate(pl.prompt_logprobs)
    ]
    res = {
        "model": args.model,
        "eager": args.eager,
        "greedy": [
            {
                "prompt": p,
                "text": o.outputs[0].text,
                "ids": list(o.outputs[0].token_ids),
            }
            for p, o in zip(PROMPTS, outs)
        ],
        "greedy_wall_s": gen_s,
        "prompt_ids": list(ids),
        "prompt_logprobs": lps,
    }
    vals = [v for v in lps if v is not None]
    res["mean_nll"] = -sum(vals) / len(vals)
    res["ppl"] = math.exp(res["mean_nll"])
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for g in res["greedy"]:
        print(repr(g["prompt"]), "->", repr(g["text"]))
    print(f"teacher-forced ppl {res['ppl']:.4f} over {len(vals)} tokens")


def compare(a_path: str, b_path: str) -> None:
    with open(a_path) as fa, open(b_path) as fb:
        a, b = json.load(fa), json.load(fb)
    assert a["prompt_ids"] == b["prompt_ids"]
    pairs = [
        (x, y)
        for x, y in zip(a["prompt_logprobs"], b["prompt_logprobs"])
        if x is not None and y is not None
    ]
    gaps = [abs(x - y) for x, y in pairs]
    print(f"ppl A {a['ppl']:.4f}  B {b['ppl']:.4f}  ratio {a['ppl'] / b['ppl']:.5f}")
    print(
        f"|dlogprob| mean {sum(gaps) / len(gaps):.5f} max {max(gaps):.5f} "
        f"over {len(gaps)} tokens"
    )
    for ga, gb in zip(a["greedy"], b["greedy"]):
        n = next(
            (i for i, (x, y) in enumerate(zip(ga["ids"], gb["ids"])) if x != y),
            min(len(ga["ids"]), len(gb["ids"])),
        )
        print(f"{ga['prompt']!r}: identical first {n}/{len(ga['ids'])} tokens")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--out")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem", type=float, default=0.35)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--compare", nargs=2)
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()

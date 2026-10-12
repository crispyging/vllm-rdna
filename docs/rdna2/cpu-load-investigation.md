# CPU load while serving on gfx1030 (investigation)

Goal: cut host CPU use during inference **without slowing inference**. Any
change must hold prefill tok/s, decode tok/s per request, TTFT and ITL (1k/512
and 16k/1k at c=1/c=8) within noise.

## What burns CPU today

Sampled with `top -H` and `py-spy dump --native` / `eu-stack` on a live
Flash-Next server (TP=4, MTP2, F&P, 16k/1k c=8) on 2026-10-12:

| Process / thread | CPU | Stack |
|---|---|---|
| GPU worker, async output copy (`WorkerAsyncOutputCopy`) | ~100 % | `copy_event.synchronize()` (`vllm/v1/worker/gpu/async_utils.py`) -> `rocr::core::BusyWaitSignal::WaitRelaxed`. The event is already `torch.cuda.Event(blocking=True)`, but the ROCm signal behind it spins anyway. |
| GPU worker, ROCm runtime thread | ~100 % | `rocr::core::Runtime::AsyncEventsLoop` polling instead of sleeping on interrupt events. |
| GPU worker, main thread | ~40 % | `shm_broadcast` `wait()` with `sched_yield` during the busy window after each step (upstream design; it sleeps on a socket once idle). |
| Engine core | ~40 % | scheduler / input preparation. |

About 2.4 cores per GPU worker, ~10 cores for TP=4, are spent waiting on
the GPU rather than working. Two of the three spinners are in the ROCm runtime's
signal handling, which points at signals not using interrupts on this stack.

## Wait microbench (one V620, 3 s GPU job, CPU while waiting)

| Wait | CPU | Busiest threads |
|---|---|---|
| blocking event / default event / stream sync | ~200 % | waiting thread 100 %, ROCr `AsyncEventsLoop` 100 % |
| `event.query()` + 50 µs sleep | ~105 % | `AsyncEventsLoop` 100 %, waiting thread 5 % |
| `event.query()` + 200 µs sleep | ~102 % | `AsyncEventsLoop` 100 %, waiting thread 3 % |

`HSA_ENABLE_INTERRUPT=1` (default) and `ROC_ACTIVE_WAIT_TIMEOUT=0` change
nothing; `HSA_ENABLE_INTERRUPT=0` makes it worse (~300 %). So the ROCm env
cannot fix it: a blocking wait spins in `BusyWaitSignal::WaitRelaxed` whatever
the flags. Polling frees the waiting thread (~1 core per GPU worker); the
runtime's `AsyncEventsLoop` keeps spinning in every mode.

## Serve A/B: polling wait (candidate 2), 2026-10-12

Baseline tree `rdna_extras@c057ee548b` + the opt-in polling wait, TP4 F&P,
V620s capped at 175 W, `VLLM_ROCM_POLL_WAIT_US=0` vs `50`, alternating. CPU is
the whole server process group averaged over the four bench cells.

| Arm | Bench-window CPU | 16k/1k c=8 prefill / decode / TTFT / ITL |
|---|---|---|
| MTP0, off (2 runs) | 902 / 906 % | 1068, 1063 / 13.30, 13.26 / 15.34, 15.42 s / 35.7 ms |
| MTP0, 50 µs (2 runs) | 826 / 825 % | 1058, 1065 / 13.03, 13.28 / 15.48, 15.39 s / 35.5 ms |
| MTP2, off | 896 % | 1470 / 13.83 / 11.14 s / 62.0 ms |
| MTP2, 50 µs | 820 % | 1176 / 14.52 / 13.93 s / 62.7 ms |

- Saves ~0.8 core (~9 %) of server CPU, not the ~1 core per worker of the
  microbench: in serving the output-copy thread only waits part of each step.
- MTP0: no measurable cost (second pair at parity on all four metrics).
- MTP2: one pair only. 16k c=8 prefill/TTFT lower with polling (1176 vs 1470,
  13.9 vs 11.1 s) but c=8 prefill varies this much run to run on MTP2 (1340-1707
  across identical bundle runs), and ITL is equal. Not proven either way.
- Verdict: keep it opt-in (default off). The bulk of the CPU (~8 cores) is the
  ROCr `AsyncEventsLoop` and the shm_broadcast busy window, which polling does not touch.

## Candidates (each needs the full A/B before adoption)

1. ~~ROCm env makes signal waits sleep~~: measured, no effect (above).
2. **Output-copy thread polls instead of blocking.** `copy_event.query()` with a
   ~50 µs sleep instead of `synchronize()` in `get_output`: frees ~1 core per
   worker in the microbench. Risk: up to ~50 µs added per step (ITL 22-60 ms);
   serve A/B with ITL first.
3. **`AsyncEventsLoop`.** Find which async handlers keep it awake (HIP callbacks,
   `hipLaunchHostFunc`, RCCL proxy) and whether fewer registered signals let it
   sleep.
4. **shm_broadcast busy window.** Shorten `busy_loop_s` on RDNA if the main
   thread's spin does not buy step latency (measure ITL).
5. **Engine core.** Profile the scheduler share (py-spy record) at c=8 for
   avoidable per-step Python work.

## Status

- [x] Live profile (above)
- [x] Single-GPU wait microbench (env: no effect; polling: -1 core per worker)
- [x] Polling prototype (candidate 2) + serve A/B: -0.8 core, MTP0 parity, MTP2 unproven -> opt-in
- [ ] Candidates 2-5

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

## Candidates (each needs the full A/B before adoption)

1. **ROCm signal waits sleep instead of spin.** Measure CPU time during a
   blocking wait under `HSA_ENABLE_INTERRUPT=1|0`, `ROC_ACTIVE_WAIT_TIMEOUT=0`
   (CLR active-wait window), and combinations (single-GPU microbench, then a
   serve A/B). Risk: wake-up latency adds to every step, so ITL is the metric to
   watch.
2. **Output-copy thread waits less often.** Poll `copy_event.query()` with a short
   sleep, or wait only when the scheduler needs the output, instead of one
   blocking sync per step.
3. **`AsyncEventsLoop`.** Find which async handlers keep it awake (HIP callbacks,
   `hipLaunchHostFunc`, RCCL proxy) and whether fewer registered signals let it
   sleep.
4. **shm_broadcast busy window.** Shorten `busy_loop_s` on RDNA if the main
   thread's spin does not buy step latency (measure ITL).
5. **Engine core.** Profile the scheduler share (py-spy record) at c=8 for
   avoidable per-step Python work.

## Status

- [x] Live profile (above)
- [ ] Single-GPU wait microbench under the env settings in (1)
- [ ] Serve A/B for whatever cuts CPU in the microbench (4 cells, ITL first)
- [ ] Candidates 2-5

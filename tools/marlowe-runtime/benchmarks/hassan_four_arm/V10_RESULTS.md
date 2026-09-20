# Frozen V10 evaluation checkpoint

Updated: **2026-09-19T20:28:00.693225-07:00 (PDT)**.

**Recommendation: preserve V10 as the candidate to polish and review. Stop runtime tuning in this evaluation.** It improves all four matched HiSparse decode cases by 10.34–14.03%, preserves Hassan’s two-stream benefit and approximately 96% waiter-overhead removal, and passes the retained correctness checks. It is not a universal nonregression result: five of 329 broader microbenchmark cells regress 2.54–3.79%. Do not treat this as unrestricted production qualification.

## HiSparse simplified GLM decode — TP4

Completed on node1 using two held four-GPU allocations (58779 C1, 58780 C4), each running stock-A/off-A/on-A/on-B/off-B/stock-B. Same retained e4b3fe5 application, model, tapes and protocol: 32K prompt, 64 warmup tokens, 192 measured intervals. V10-off is the same candidate bytes with features disabled; it is not stock.

All values ms/token; six retained trials from two independent model starts. Tables use pooled medians; per-resident medians are shown separately.

| Case | Stock | V10 off | V10 on | On vs stock | V10 range |
|---|---:|---:|---:|---:|---:|
| C1 prefetch off | 17.040680 | 17.130802 | 15.044653 | -11.71% | 15.031517–15.063908 |
| C1 prefetch on | 17.631078 | 17.368385 | 15.157916 | -14.03% | 15.127131–15.164286 |
| C4 prefetch off | 20.106943 | 20.085779 | 18.027149 | -10.34% | 17.982687–18.066943 |
| C4 prefetch on | 20.279640 | 19.958310 | 17.990133 | -11.29% | 17.951810–18.082042 |

| Case | Runtime | Resident A median | Resident B median | B vs A |
|---|---|---:|---:|---:|
| C1 prefetch off | stock | 17.037053 | 17.049431 | +0.07% |
| C1 prefetch off | off | 17.124806 | 17.154097 | +0.17% |
| C1 prefetch off | on | 15.060090 | 15.031944 | -0.19% |
| C1 prefetch on | stock | 17.673653 | 17.614437 | -0.34% |
| C1 prefetch on | off | 17.369976 | 17.363215 | -0.04% |
| C1 prefetch on | on | 15.164268 | 15.127351 | -0.24% |
| C4 prefetch off | stock | 20.117372 | 20.093808 | -0.12% |
| C4 prefetch off | off | 20.146182 | 20.070645 | -0.37% |
| C4 prefetch off | on | 18.002524 | 18.054471 | +0.29% |
| C4 prefetch on | stock | 20.350111 | 20.261999 | -0.43% |
| C4 prefetch on | off | 19.984889 | 19.947341 | -0.19% |
| C4 prefetch on | on | 17.982745 | 18.012975 | +0.17% |

Both remote primary audits and independent local mirror audits passed: **72 timing trials, 72 natural retrieval checks, 72 process identities and 72 mapped-library records** in total. The retrieval checks are bounded model-correctness probes, not comprehensive accuracy qualification. All trials were retained; exact hashes, controls, token tapes, application recipe and distinct worker maps were checked.

V10-on repeat medians differ by less than 0.3%. Prefetch-on versus off is +0.75% at C1 and −0.21% at C4; these data do not establish why.

## Earlier best and fully resident references

Historical comparisons below are unmatched: earlier runtime/application bundles and/or nodes differ. They show distance to prior results, not causal runtime effects. Earlier experimental offload medians were C1 off/on 15.033/14.909 and C4 off/on 18.090/17.800 ms/token. V10 is near that earlier bundle, not consistently faster than it.

Fully resident KV (HiSparse disabled) was **13.520 ms/token C1** and **16.038 C4**, from the September 15 integrated application on node3, commit17498a1, one window per point. Current V10 uses retained e4b3fe5 on node1. Both use TP4 GLM5.2 FP8/BF16 KV and the same stated prompt/warmup/window sizes. Resident was not rerun with V10.

| Case | Resident historical | V10 HiSparse | Gap | Above resident |
|---|---:|---:|---:|---:|
| C1 prefetch off | 13.520 | 15.045 | +1.525 ms/token | +11.28% |
| C1 prefetch on | 13.520 | 15.158 | +1.638 ms/token | +12.11% |
| C4 prefetch off | 16.038 | 18.027 | +1.989 ms/token | +12.40% |
| C4 prefetch on | 16.038 | 17.990 | +1.952 ms/token | +12.17% |

Resident source: `/home/sashawork/dev/hisparse-decode-20260914/REPORT.md`.

## Completed microbenchmarks and compatibility


| Benchmark | Stock | Earlier matched runtime | V10 off | V10 enabled | Result |
|---|---:|---:|---:|---:|---|
| Hassan two streams, group 1 | 14.139 µs | 11.573 µs | 21.539 µs | **10.071 µs** | 28.77% faster vs stock |
| Hassan two streams, group 50 | 10.084 µs | 7.293 µs | 10.048 µs | **7.208 µs** | 28.52% faster vs stock |
| Hassan serial, group 1 | 13.575 µs | 12.479 µs | 13.535 µs | **12.542 µs** | 7.60% faster vs stock; 0.51% slower vs earlier |
| Hassan serial, group 50 | 8.940 µs | 8.901 µs | 8.815 µs | 8.918 µs | Near neutral vs stock |
| Added waiter overhead | 2097.640 µs | — | — | **86.005 µs** | 95.90% removed; added wait cost, not end-to-end latency |
| Broad micros, 329 cells | reference | See family table | See family table | median **−0.13% vs stock** | 60 wins >2%, 264 neutral within ±2%, 5 regressions >2% |

Broad-micro regressions remain: two dispatch cases +3.79%/+2.54%, three pipeline cases +2.71%/+3.11%/+3.43%. No runtime tuning is being attempted during qualification. Hassan structural evidence supports genuine two-stream execution; maximum occupancy and a firmware cause are not established.

Correctness: 90 single-GPU lifecycle/PyTorch checks passed (58139); stock/off/on two-rank graph replay and NCCL all-reduce passed (58289). Broad micro matrix 58498 and Hassan 58500 passed their independent audits.

## Detailed microbenchmark families


Job 58498 covers 329 cells with four rotated stock/previous/V10-off/V10-on rounds. An independent local audit passed 1,463 assertions over 180 process receipts and 46,912 timing rows, including exact controls, mapped hashes, and mirrored output checksums.

| Family | Cells | V10 vs stock median [range] | V10 vs previous median | V10 vs off median | Wins / neutral / regressions vs stock | Median four-round span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention_fetch | 120 | -0.11% [-8.29, +1.69] | +0.01% | -0.06% | 20 / 100 / 0 | 0.73 pp |
| dispatch | 42 | +0.05% [-9.43, +3.79] | +0.07% | -0.04% | 3 / 37 / 2 | 1.54 pp |
| experts | 52 | -0.78% [-34.98, +0.66] | -0.11% | -0.63% | 12 / 40 / 0 | 1.72 pp |
| fanout_q4 | 6 | -18.29% [-65.20, +0.08] | -0.03% | -18.28% | 3 / 3 / 0 | 2.66 pp |
| fanout_q8 | 6 | -18.35% [-65.11, +0.07] | -0.06% | -18.35% | 3 / 3 / 0 | 0.24 pp |
| grouped_s1 | 12 | -0.53% [-4.06, +0.26] | +0.00% | -0.54% | 3 / 9 / 0 | 0.16 pp |
| grouped_s22 | 12 | -6.16% [-8.82, -5.41] | -0.66% | -6.55% | 12 / 0 / 0 | 0.74 pp |
| mixed | 16 | -0.06% [-1.14, +1.85] | +0.16% | -0.10% | 0 / 16 / 0 | 1.31 pp |
| pipeline | 36 | -0.03% [-7.09, +3.42] | +0.01% | +0.06% | 3 / 30 / 3 | 0.62 pp |
| queued | 24 | +0.03% [-0.65, +0.54] | -0.12% | -0.06% | 0 / 24 / 0 | 0.69 pp |
| waiter | 3 | -0.09% [-38.27, -0.08] | -0.22% | +0.09% | 1 / 2 / 0 | 0.76 pp |

The five >2% regressions versus stock are:

- Dispatch `256×1024`, one graph, eager serial: 353.171 → 366.562 µs (+3.79%).
- Dispatch `256×1024`, one graph, eager wide: 181.306 → 185.906 µs (+2.54%).
- Pipeline graph, two streams, `h128-l8-miss128`: 2167.881 → 2226.623 µs (+2.71%).
- Pipeline graph, two streams, `h32-l32-miss128`: 8634.832 → 8903.771 µs (+3.12%).
- Pipeline graph, two streams, `h32-l8-miss32`: 1701.754 → 1760.034 µs (+3.43%).


## Interpretation and qualification limits

Established: the frozen candidate improves Hassan and retained HiSparse versus stock while retaining the large waiter-overhead reduction. The matched micros include four rotated stock/previous/off/on rounds. Hassan structural evidence supports genuine parallel streams; it does not prove maximum occupancy or a firmware cause. Five microbenchmark regressions remain.

Hypothesis only: fixed graph scheduling/retirement work may outweigh saved waits in the regressing topologies. This evaluation does not establish the mechanism; no runtime tuning or gap investigation was performed.

Standard non-HiSparse GLM C1/C4/C16 TP8/EP8 900-second runs were excluded following the user’s explicit scope correction to simplified TP4 decode. **No result or qualification claim is made for those workloads.** The historical resident reference is not a substitute. Broader production workloads, sustained service behavior and comprehensive model accuracy remain unqualified.

## Preservation, packaging and rollback

Clean review PR: https://github.com/MarloweAI/clr/pull/3, branch `marlowe/frozen-v10-evaluation-clean`. Runtime and Hassan reproducer are separate commits. Runtime source is frozen; subsequent documentation commits do not change evaluated bytes.

Frozen SHA-256:

- HIP: `857a3d8d714f057cb04bc5025434200eaa5a3c6a31ad8224d94e59d13cd241a2`
- HSA: `2899f94063127a3c6d0bbba0e5c5c54cab3256499f2f0183f6a53631b5fdad12`

Package retained under `amd-runtime-production/iterations/v10-e2e-qualification-20260919/package/` in both `/home/sashawork/dev` (GCP) and `/workspace/home/sasha` (cluster). `manifest.json` records aliases and hashes. Enablement and rollback are documented in PR3 `tools/marlowe-runtime/benchmarks/hassan_four_arm/V10_EVALUATION.md`.

Use isolated candidate workers. The evaluated disposable container redirects PyTorch-bundled HIP/HSA paths to frozen libraries for candidate-off/on and restores hash-checked stock paths for stock. Preload alone can leave duplicate runtimes mapped and is insufficient. Verify each worker’s mapped hashes and controls. Roll back by restarting with stock bundled paths and removing candidate preloads/controls; feature-off is not rollback. No system runtime replacement or deployment was performed.

Mount, startup receipt and bundled-library fixes were harness/packaging only. Earlier failed attempts remain preserved and excluded from the balanced matrix. Any source change or rebuild is a new candidate requiring validation.

## Evidence

Local root: `/home/sashawork/dev/amd-runtime-production/iterations/v10-e2e-qualification-20260919/`. Cluster mirror root: `/workspace/home/sasha/amd-runtime-production/iterations/v10-e2e-qualification-20260919/`.

- `HISPARSE-FINAL-SUMMARY.json`: all trials, pooled and per-resident medians.
- `hs-full-matrix/hs-c1-bridge-j58779/` and `hs-c4-bridge-j58780/`: raw cohort mirrors, primary/mirror audits and MIRROR_COMPLETE markers.
- Broad micros job58498; Hassan job58500; compatibility jobs58139 and58289.
- `LIVE-PROGRESS.md`: PDT completion timeline and resource history.

The scoped TP4/microbenchmark evaluation is complete; standard TP8 production qualification remains unmeasured.

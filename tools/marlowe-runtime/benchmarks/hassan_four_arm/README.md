# Native event wait × indexer scheduling reproducer

This package contains the exact six-file SGLang change, retained serving runner/client and statistical code, a portable Slurm adapter, and the completed experiment's compact results. The adapter was tested on CPU during packaging; it has **not** been rerun on GPUs. The archived numbers below are actual serving measurements.

| Arm | `GPU_NATIVE_EVENT_WAIT` | `SGLANG_ROCM_INDEXER_PROJECTION_SCHEDULE` | Archived matched mean (ms) |
| --- | --- | --- | ---: |
| A | `0` | `serial` | 3.344462 |
| B | `0` | `events` | 3.540069 |
| C | `1` | `serial` | 3.342588 |
| D | `1` | `events` | 3.637258 |

These means use seven complete matched blocks, n = 7 and df = 6. D was 8.82% slower than C and 2.75% slower than B at the serving-cadence endpoint. This is a descriptive result, with `accepted=false`; it is not a production recommendation or an attribution of the entire difference to native wait overhead.

## Inputs and site assumptions

The launcher requires Linux Slurm with Pyxis/Enroot, `srun`, `scontrol`, `squeue`, `sacct`, `scancel`, Git and Python 3.10+ on the login and compute hosts. Run the controller from a login shell outside a Slurm step. Analysis/tests use Python 3.12+ and the pinned packages in `requirements-analysis.txt`.

The recorded node has eight MI355X (`gfx950`) GPUs and 64 assigned CPU threads spanning two sockets: CPUs 0–31 and 64–95. Actual GPU visibility and that affinity are checked. The allocation must have one node, exactly eight GPUs, 64 CPUs and a time limit no longer than six hours. It must be dedicated to this campaign, with no other work in its steps. No control hostname or username is embedded here.

Obtain the following existing assets from their authorized custodian. They are **not distributed in this repository**. The launcher checks the complete SHA-256 identities in [contracts/configuration.json](contracts/configuration.json), [contracts/runtime.json](contracts/runtime.json), and [contracts/model-files.json](contracts/model-files.json).

| Required input | Identity / layout |
| --- | --- |
| SGLang Git object database | Base `fb91baedab3e1de668d6e8f391ccd81cab9e9acd` from `sgl-project/sglang`; the exact patch is included |
| SGLang ROCm SQSH | `43b9e7ae1e56863cd48be7af67958b8dd7ff833a907f13763883cd668a13ce06` |
| Source-built native runtime archive | `cf49bdef0a82a270d61cb2d068eff986a71cee2da71e5c9727cfdbced6179531` |
| Extracted runtime release directory | Its `manifest.json`, `lib/`, `run`, `verify.py` and licenses; all libraries, aliases and launch files must match the archive contract |
| Frozen warm cache archive | `a6ad2ff2c3ecef82a3649d5d9a1596805256c47d7f95d00734662704985a61c5`; 8,774,277,120 bytes; top-level `seed/` |
| Retained L10 checkpoint | `amd/GLM-5.2-MXFP4`, snapshot `7b7a06362a78f379a7b4f559640e90422937cebe`, with the exact retained configuration/file bytes in the model manifest |

The full upstream checkpoint alone is insufficient: the retained configuration has ten target layers and one draft layer. Per target rank the loaded weight count is 5,947,301,024 bytes; invoked indexers are layers 0, 1, 2 and 6. Draft decoding uses `EagerRunner`. HiCache remains enabled. Model code is trusted through the frozen `--trust-remote-code` configuration; inspect the supplied checkpoint before use.

The image provides the recorded Torch/ROCm/AITER stack, `/opt/rocm`, and `/sgl-workspace/aiter/aiter/jit`. The cache policy binds a fresh writable copy of the same seed for each group, including that AITER directory. The startup host-pointer compatibility bridge is included unchanged. Different images, cache seeds, runtime binaries, projection dispatches or checkpoint files fail closed; replacing a hash is a new experiment requiring qualification.

The optional [runtime-recipe](runtime-recipe/) records the CLR/HIP source locks and original CPU build recipe. A rebuild is a **new, initially unqualified artifact**, even from identical source. It is not automatically interchangeable with the measured archive. The reproducer never installs or overwrites a system runtime.

## Stage and run a new campaign

1. Put the repository and all inputs on one filesystem visible at the same absolute paths from login and compute nodes. Create a fresh patched worktree:

   ```bash
   python3 reproduce.py prepare-source \
     --base-repo /shared/sglang-object-database \
     --destination /shared/patched-sglang
   ```

   This applies [patches/sglang-indexer-events.patch](patches/sglang-indexer-events.patch) at the exact base and verifies all six resulting hashes. The patched worktree stays at that base with exactly six uncommitted changes; do not commit or add unrelated files there before running.

2. Copy `site.example.json` to `site.json` and fill **every** input. `shared_root` is the common bind mount; every path must be below it. `output_root` must not exist and must be disjoint from source, runtime and model trees. The runtime directory must already be extracted. No download, artifact substitution, inferred resume or overwrite occurs.

   ```bash
   python3 reproduce.py plan --site site.json
   ```

3. From the login shell, submit the included idle allocation with the site's partition, job name matching `site.json`, and a fresh log path. The example requests the required two-socket TP8 placement without `--exclusive`:

   ```bash
   sbatch --parsable --partition=YOUR_PARTITION \
     --job-name=native-runtime-reproducer \
     --output=/shared/allocation-%j.log allocation.sbatch
   ```

   Wait until that explicit job is `RUNNING`. Then, from the same login environment, run:

   ```bash
   python3 reproduce.py run --site site.json --job-id YOUR_JOB_ID
   ```

   The controller checks the job's user, name, partition and resource bounds, stages and hashes the payload, atomically claims the job on its node, verifies external assets, prepares the node-local seed, and starts fresh worker groups. It releases **only that explicitly claimed job** at completion or failure. Keep the login controller running; monitor its output and the staged step/client logs. It prints validity and duration progress, leaving treatment contrasts for terminal analysis.

The prospective protocol is fixed **before A/A**: eight A/A groups, then eight four-arm blocks ordered `ABDC, BCAD, CDBA, DACB` twice. All 40 groups use the same 510-second server-lifetime cap, 495-second work budget and 15-second cleanup reserve. Cache restoration and attestation precede that server lifetime but count against the six-hour allocation. Each group has 33 warmup requests and 83 measured requests, concurrency 1, seed 20260909, random input/output length arguments 8192/1024 and range ratio 0.8, and at least 60 seconds of measured serving. Full resolved server arguments and environment are retained in [contracts/server-argv.json](contracts/server-argv.json) and [harness/common_env.sh](harness/common_env.sh).

A/A freezes `delta = max(0.005 ms, 0.5% of the A/A mean, 2 × the maximum absolute paired A/A difference)`. A valid calibration above 2% is labeled unresolved noise; the unchanged fixed slots may continue descriptively. A malformed/invalid group stops the campaign: no replacement, count change, adaptive cap, recalibration or automatic resume. The 1,170-second admission guard can stop before all slots finish. The guard reserves 690 seconds for the group, 60 seconds for receipt visibility, up to 362 seconds for terminal health/cache/recheck/cancel/queue/accounting operations, and 58 seconds of margin. Six hours is a resource ceiling, not a promise that another site's startup costs fit.

The endpoint is the arithmetic mean of target-runner cadence intervals in **one continuous 83-request measured phase**. Only the first interval crossing the warmup/measurement boundary is excluded. Request transitions and pauses remain included. It is not isolated graph replay latency or kernel execution time.

## Analyze

Install the CPU analysis environment, then regenerate compact archived statistics without external assets:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python analyze.py archived --output archived-regenerated.json
```

The compact check recalculates the archived means and paired intervals from the retained group summaries. For a full raw reduction, obtain the optional evidence snapshot and use:

```bash
.venv/bin/python analyze.py archived --raw /path/to/evidence-snapshot \
  --output archived-from-raw.json
```

For a new campaign, wait for its **own terminal receipt**, then run:

```bash
.venv/bin/python analyze.py prospective --run /shared/new-run \
  --output prospective-results.json
```

Analysis is read-only on raw data and refuses an existing output filename. Prospective reduction requires all eight valid A/A groups and their frozen calibration. It retains missing/invalid slots and partial blocks; main comparisons use only complete matched quartets, with actual n/df and no confidence interval for n < 2. All results remain descriptive with `accepted=false`. A setup failure before calibration has only diagnostic/cleanup receipts, not a valid numerical result.

## Historical result and interpretation

[expected/results-descriptive.json](expected/results-descriptive.json) retains the historical result: all-valid treatment counts A/B/C/D = 8/7/8/8. Original `block-00-B` failed at the work deadline after 83 HTTP completions while waiting for client exit, before collecting the primary endpoint and postchecks. It was never replaced or relabeled valid. The original 480-second cap became 510 seconds prospectively for the remaining original slots. Main comparisons exclude all of partial block 00 and use complete blocks 01–07; all valid partial results remain separately represented. **New campaigns do not inherit that invalid slot or amendment.**

The archived frozen delta is 0.0534982894 ms. C−D is −0.294670 ms (98.333% CI [−0.304768, −0.284572]); B−D is −0.097189 ms ([−0.116350, −0.078028]); the interaction `(B−A)−(D−C)` is −0.099063 ms ([−0.122646, −0.075480]). A−D is −0.292796 ms (95% CI [−0.301611, −0.283981]). Percentages in the result name their matched control denominator. Paired-t intervals assume independent, approximately normal block differences; n = 7 is small.

Qualification established tested event dependencies and finite complete serving under frozen empirical Q/K bounds, **not a formal error guarantee**. Original full-model bitwise equivalence failed because serial projection repeats already differed. Q normalized RMS/max-absolute-to-reference-RMS caps are 0.0067/0.20; K caps are 0.012/0.15. Downstream checks used identical injected Q/K, exact invariants, strict mathematical top-k validity at exact cutoff ties, changing-input stale witnesses and a rejected delayed/missing-join control. Exact-cutoff-valid choices can select different K/V populations; full-model numerical equivalence is not established.

Acceptance and output contents varied. Expert routing was not directly counted, so identical routed compute is not established. D's maximum cadence count exceeded the A/A maximum by 6; its minimum mean acceptance was below the A/A minimum by 0.000078834. Whole-result changes cannot be assigned entirely to native wait overhead. Separate diagnostic traces observed native packet engagement (C: 0, D: 9 per rank), separate queues, and **0/32 positive overlaps of the selected BF16 Q/K hgemm dispatch timestamp intervals**. That is not a whole-projection overlap census or proof of structural impossibility. One approved read-only CPU progress/clock step overlapped historical D1; its receipt exists but the original inline command was not retained. No profiled latency entered the primary measurement. No full-depth, production or deployment inference is supported.

## Cleanup and recovery

Output roots and per-job node claims are collision-safe. A second invocation with the same job and another output root loses the atomic claim and **cannot cancel the first campaign**. Claims are never automatically reclaimed. The small node-local `...-claim` directory remains as an ownership tombstone; only the matching controller can use it. If a controller is lost, confirm the recorded job and **all its steps** are terminal and inspect the matching UID/job/output/claim receipts before an operator removes that exact stale claim. Do not remove a live claim to start another controller.

The controller attempts bounded worker/VRAM health, ownership-checked cache removal and exact-job cancellation independently. A final node recheck timeout cannot bypass cancellation after the initial verified claim. `release.json`, `accounting.txt`, `health-final.*` and `cache-cleanup.json` retain success or failure. Nonzero cleanup, missing census, live/ambiguous PIDs, malformed health or unverified cache absence cannot produce a successful campaign verdict. Raw device-used bytes are retained separately from zero allocated-percent and KFD evidence; zero percent is not claimed to mean zero raw bytes. On incomplete worker evidence, cache deletion fails closed and the exact job is still released; use the retained ownership receipts for site-operator recovery. Preserve all raw outputs. No shared cache or unrelated job is cleaned.

## Evidence and CPU checks

The optional raw snapshot has 2,312 files totaling 2,365,227,522 bytes. [contracts/evidence-files.sha256](contracts/evidence-files.sha256) identifies each relative file without embedding a private storage URL. Obtain the snapshot separately and verify it from its root with `sha256sum --check /path/to/reproducer/contracts/evidence-files.sha256`. The original execution manifest and additive cap amendment are in `contracts/historical-*.json`; original and portable result hashes are in [expected/provenance.json](expected/provenance.json). Only absolute group-path prefixes were normalized to `raw/`; numerical values and historical invalidity are unchanged.

```bash
.venv/bin/python -m unittest discover -s tests -v
SGLANG_BASE_REPO=/path/to/sglang-object-database \
  .venv/bin/python -m unittest discover -s tests -v
```

The second form also exercises applying all six source changes to the exact base. Tests cover atomic competing claims, losing-claim cancellation refusal, partial preparation, nonzero cleanup, health/recheck timeouts, worker step/census corruption, a real work-deadline signal, runtime release layout, archived analysis and active-campaign refusal. This validates the adapter on CPU, not a new GPU qualification. The included SGLang/client source is Apache-2.0; runtime recipe/vendor notices remain with their source. No model weights, credentials, confidential source decks or raw logs are included.

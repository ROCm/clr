# Reproducer implementation plan

Status: complete. The implementation, CPU checks and independent reproducibility review passed.

Goal: preserve the completed four-arm evidence and provide a prospective, explicitly configured rerun without private control-host dependencies.

- [x] Inventory and hash the exact source, retained runner/client and result contracts.
- [x] Add collision-safe site configuration, staging, owned-job execution and cleanup.
- [x] Keep historical reduction separate from a prospective uniform-cap eight-block protocol.
- [x] Exercise CPU failure paths, patch application and archived numerical regeneration.
- [x] Complete independent reproducibility review and create one local commit.

The original SGLang change and GPU compute are fixed. The portable adapter may relocate paths, but measurement counts, cadence extraction, timeouts, cache bytes and arm order stay explicit. No GPU run, artifact acquisition, model weights or raw logs are part of this packaging task.

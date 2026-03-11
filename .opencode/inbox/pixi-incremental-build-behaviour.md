# Observation: pixi build/test pipeline is fast and incremental

**Observed:** 2026-03-10, session remove-allocator-registry-singleton, subtask 06

The `pixi run build dev` step correctly detects when build artifacts are up to date and exits cleanly with "no work to do" — no need to force a clean rebuild after header-only changes that were already committed. This makes the build-test loop fast (configure ~0.3s, build nearly instant, 64 tests in ~12s).

**Pattern:** After header-only refactors, if all changes are committed before running the test pipeline, the build step will already be current from the prior incremental build. Only force a clean if you suspect stale objects from a branch switch or major rename.

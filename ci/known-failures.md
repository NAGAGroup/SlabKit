# Known failures excluded in CI

Tests excluded by exact name on linux-aarch64, osx-arm64, win-arm64 and Metal. Nothing is excluded on linux-64 or win-64.
The names live in `ci/exclude/<platform>-<backend>.txt`; each is listed here with its failure text.

## osx-arm64, Metal (informational)

`a task on the unordered stream that depends on a container's write list waits for a writer on that stream`
(`tests/test_unordered_queue.cpp:368`). The test launches a kernel that spins until the host opens a gate. On Metal the
spinning command buffer hits the GPU watchdog and the process is terminated, which ends the whole run (run 39211549744):

```
tests/test_unordered_queue.cpp:397: FAILED:
  REQUIRE( wait_done(e) )
with expansion:
  false
[AdaptiveCpp Error] from ../work/AdaptiveCpp/src/runtime/metal/metal_queue.cpp:370 @ operator()(): metal: Command buffer failed: Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)
AdaptiveCpp has caught the following unhandled asynchronous errors: ...
The application will now be terminated.
libc++abi: terminating
```

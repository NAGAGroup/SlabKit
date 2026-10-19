# Known failures excluded in CI

Tests excluded by exact name on linux-aarch64, osx-arm64, win-arm64 and Metal. Nothing is excluded on linux-64 or win-64.
The names live in `ci/exclude/<platform>-<backend>.txt`; each is listed here with its failure text.

## osx-arm64, Metal (informational)

Eighteen tests launch a kernel that spins until the host opens a gate (`Gate` / `WritingGate` in the test sources). On Metal a
spinning command buffer hits the GPU watchdog and AdaptiveCpp terminates the whole process, which ends the run, so the tests
after it never execute. Observed individually in two of them (runs 39211549744 and 39214204546):

```
tests/test_unordered_queue.cpp:397: FAILED:   REQUIRE( wait_done(e) ) with expansion: false
tests/test_usm_vector.cpp:672: FAILED:        REQUIRE( wait_done(e) ) with expansion: false
[AdaptiveCpp Error] from ../work/AdaptiveCpp/src/runtime/metal/metal_queue.cpp:370 @ operator()(): metal: Command buffer failed: Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)
AdaptiveCpp has caught the following unhandled asynchronous errors: ...
The application will now be terminated.
libc++abi: terminating
```

The other sixteen are excluded on the same mechanism, found by scanning the sources for gate use, not observed one by one:

- `a kernel with a dependency does not run until the event completes`
- `a kernel without dependencies is not ordered after earlier work on the stream`
- `an async_host task runs after its dependency, and its exception surfaces`
- `wait returns only after all submitted work has finished`
- `usm_vector::read_deps includes registered writes, and a task depending on it waits`
- `a task on the unordered stream that depends on a container's write list waits for a writer on that stream`
- `outside readers do not wait on each other`
- `a writer waits for an in-flight reader`
- `outside writes do not wait on each other`
- `a registered write delays a later fill, resize or copy-assign`
- `assign orders against both containers`
- `an event list is handed out whole: every registered event is in it`
- `a free waits on both lists`
- `sync waits on both lists`
- `a copy out to the host waits on the write list and registers a read`
- `the event lists travel with a move`
- `a registration that makes a list longer than 8 events drops the completed ones`
- `the extra-dependency overloads order after the given events`

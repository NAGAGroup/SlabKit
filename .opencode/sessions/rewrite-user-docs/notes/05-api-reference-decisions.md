# Subtask 05 Notes — api-reference.md

## Decisions

- **DeviceMemoryPool not documented** — internal type; only noted as return type of `pool_at()`/`pool_for()` with a note that it's an implementation detail.
- **DeviceScratchBuffer** — documented with deprecated notice, one-liner pointing to `DeviceScratchMemory`.
- **Transfer tiers** — organized into 4 explicit tiers (contiguous async / strided staging / strided D2D / blocking sync) to make the overload set scannable.
- **Implicit conversion example** — cannot use owning `DeviceArray arr` inside a kernel lambda via implicit conversion (lambda captures would capture a deleted copy). Fixed to use a named function accepting `DeviceView` explicitly.
- **MirroredArray kernel example** — `sycl::depends_on` is not a valid `parallel_for` parameter in the USM API pattern. Fixed to use in-order queue and sequential `.wait()` calls.
- **malloc_device kernel example** — corrected to use `sycl::range<1>{N}` and `sycl::id<1>` in the parallel_for lambda, not `size_t`.

## Corrections Made by HW After DocWriter Draft
1. Implicit conversion kernel example rewritten
2. MirroredArray example: removed `sycl::depends_on`, switched to in-order queue
3. `malloc_device` example: fixed lambda signature to `sycl::id<1>`

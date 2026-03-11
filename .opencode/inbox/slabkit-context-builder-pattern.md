# SlabKit: Context Builder Pattern Observation

**Date:** 2026-03-10  
**Session:** remove-allocator-registry-singleton

## Pattern
`SlabKitContextBuilder::build()` auto-creates `sycl::context` objects — callers never construct
them manually. This means any code that previously did:

```cpp
SlabContextBuilder().contexts({ctx1, ctx2}).devices({dev1, dev2}).build()
```

Must be updated to:

```cpp
SlabKitContextBuilder().devices({dev1, dev2}).build()
```

The contexts are created automatically per platform group from the device list.

## Why `new` instead of `make_shared`
`SlabKitContext` uses `enable_shared_from_this`. Using `make_shared` with a move-constructed
temporary does NOT register the shared_ptr control block properly. Must use:
```cpp
std::shared_ptr<SlabKitContext>(new SlabKitContext(std::move(result)))
```
This is intentional and must not be "fixed" to `make_shared`.

## Test file impact
All test files using `SlabContextBuilder` with `.contexts(...)` need updating (subtask 05).

<!-- Context: project-intelligence/technical | Priority: critical | Version: 0.1 | Updated: 2026-03-06 -->

# Technical Domain — SlabKit

> ⚠️ **DRAFT — Cross-check before use**
> This context reflects an early snapshot of a greenfield project. Details
> (abstractions, allocator design, API shapes) may shift as the codebase
> solidifies. **Before relying on anything here, scan `include/slabkit/`
> headers and verify it matches. If you spot differences, recommend running
> `/add-context --update` before proceeding.**

**Purpose**: Tech stack, architecture, and development patterns for SlabKit —
a focused C++20/SYCL USM abstraction library.
**Scope**: Refinement, testing, and CMake scaffolding of an existing near-complete design.
**Last Updated**: 2026-03-06

---

## Quick Reference
**Update Triggers**: New abstractions | Build system changes | Allocator strategy changes | Headers diverge from context
**Audience**: Developers, AI agents

---

## Primary Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Language | C++20 | `CXX_EXTENSIONS OFF` (no GNU extensions) |
| SYCL runtime | AdaptiveCpp (acpp-toolchain) | Installed via pixi |
| SYCL target | `generic` (SSCP) | **Only supported target** — IR lowered at runtime to CUDA / Level-Zero / OpenCL / ROCm / OMP |
| Build system | CMake + Ninja | Configured via CMakePresets.json |
| Env manager | pixi | Manages AdaptiveCpp, CMake, Ninja, Catch2, Nushell |
| Compiler | clang++ / clang | Provided by acpp-toolchain conda package |
| Memory backend | custom (host-metadata only) | `DeviceMemoryPool` (two-level bitmap) + `DeviceScratchMemory` (bump allocator) over USM slab |
| Testing | Catch2 v3 | |
| Scripts | Nushell (.nu) | `configure.nu`, `build.nu`, `test.nu` |

---

## Architecture

**Design philosophy**: Thin RAII wrappers + memory pools over raw SYCL USM.
Explicitly avoids `sycl::buffer` — all memory is USM device memory.

**Memory hierarchy**:
```
sycl::malloc_device  ←  one slab per device (60% of device global mem)
  └─ DeviceMemoryPool    (host-metadata two-level bitmap pool — node alloc + array alloc)
  └─ DeviceScratchMemory (host-metadata bump allocator — scratch/bulk-unwind)
```

**Global state**: `AllocatorRegistry` is a Meyers singleton (`static` local in `get()`).
Lazy-initialized on first allocation per device. First-allocation cost is acceptable.

---

## Core Abstractions

> ⚠️ **Draft** — verify against `include/slabkit/` before use.

| Type | Owns memory? | Copyable | Purpose |
|------|-------------|----------|---------|
| `DeviceView<T,N>` | No | Yes (trivially) | Non-owning contiguous view, device-copyable |
| `DeviceSubView<T,N>` | No | Yes (trivially) | Non-owning strided view, device-copyable |
| `DeviceArray<T,N>` | Yes (shared_ptr) | Yes (shared ownership) | RAII owning array, constructed via factory |
| `MirroredArray<T,N>` | Yes (both) | Yes (device shared, host deep-copied) | Paired device+host storage with sync methods |

**Factory pattern** — owning types constructed via free functions, not public constructors:
```cpp
auto arr     = make_device_array<float, 2>(sycl::range<2>{H, W}, dev);  // dev is sycl::device
auto scratch = make_scratch_memory(dev, 4 * 1024 * 1024);
auto q       = make_queue(dev);
```

**Kernel capture pattern** — `DeviceView` and `DeviceSubView` are trivially copyable,
safe to capture by value into SYCL kernels:
```cpp
q.parallel_for(range, [result=result_arr.view(), input=input_arr.view()](auto id) {
    result[id] = f(input[id]);
});
// Subviews work identically:
auto sv = arr.subview(std::array<size_t,2>{r0,c0}, sycl::range<2>{H,W});
q.parallel_for(sv.get_range(), [sv](sycl::id<2> id) { sv[id] = ...; });
```

**Transfers** — async, return `sycl::event`, accept `deps` chain:
```cpp
sycl::event e = copy_to_host(arr, host_span, q, {dep_event});
sycl::event e = copy_to_device(host_span, arr, q, {dep_event});
```

---

## Naming Conventions

| Kind | Convention | Example |
|------|-----------|---------|
| Files | `snake_case.hpp` | `array.hpp`, `memory.hpp` |
| Types / classes | `PascalCase` | `DeviceArray`, `MirroredArray`, `FreeBlock` |
| Concepts | `PascalCase` | `DeviceViewLike` |
| Free functions | `snake_case` | `make_device_array`, `copy_to_host` |
| Member variables | `snake_case_` (trailing `_`) | `ptr_`, `range_`, `page_size_` |
| Template params | Single-letter conventional | `T`, `N` |

---

## Code Standards

- Header-only library — all implementation in `include/slabkit/`
- C++20 concepts for template constraints (`DeviceViewLike`)
- RAII + move-only for all owning types — no copy constructors on `DeviceArray`
- Factory functions instead of public constructors for owning types
- Async-first transfers — always return `sycl::event`, accept `deps` vector
- **No `sycl::buffer`** — USM only, always
- Nushell scripts for all build tasks (not Makefile / bash)
- Error handling strategy: **TBD**

---

## Correctness Invariants

- Subview bounds checked: `offsets[d] + sub_range[d] <= range_[d]`
- No double-free: `page_map_` assertion in `deallocate_node`
- `device_index` validated against `sycl::device::get_devices()` on first use

---

## Build Presets

Tests are **disabled by default** (`SLABKIT_BUILD_TESTS=OFF`). Use the `dev` or
`dev-debug` presets for development work that requires tests.

| Preset | Build dir | Build type | Tests | Notes |
|--------|-----------|-----------|-------|-------|
| `default` | `build/default` | Debug | OFF | Library only |
| `release` | `build/release` | Release | OFF | Library only |
| `dev` | `build/dev` | Debug | ON | **Standard dev workflow** |
| `dev-debug` | `build/dev-debug` | RelWithDebInfo | ON | Optimised + debug info |

**Always use `dev` or `dev-debug` when building or running tests:**

```nu
pixi run configure dev        # configure dev preset
pixi run build dev            # build
pixi run test dev             # run tests

pixi run configure dev-debug  # configure dev-debug preset
pixi run build dev-debug
pixi run test dev-debug
```

---

## 📂 Codebase References

| Context | File |
|---------|------|
| Core abstractions | `include/slabkit/array.hpp` |
| Memory / allocator | `include/slabkit/memory.hpp` |
| Build presets | `CMakePresets.json` |
| Env + deps | `pixi.toml` |
| Build scripts | `scripts/configure.nu`, `scripts/build.nu`, `scripts/test.nu` |

---

## Related Files
- `project-intelligence/navigation.md`

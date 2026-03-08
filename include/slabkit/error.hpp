#pragma once

// =============================================================================
// slabkit/error.hpp — AllocationError exception type
// =============================================================================
//
// Exception type for memory allocation failures in SlabKit.
// Can be thrown by DeviceMemoryPool, DeviceScratchMemory, or user code.
//

#include <exception>
#include <string>

namespace slabkit {

// =============================================================================
// AllocationError — exception for memory allocation failures
// =============================================================================

class AllocationError : public std::exception {
public:
    enum class Reason {
        PoolExhausted,  // no more free slots in pool
        Fragmented,     // unable to allocate contiguous span
        ScratchFull,    // scratch allocator exhausted
    };

    explicit AllocationError(Reason r, const char* msg = "")
        : reason_(r), msg_(msg) {}

    Reason reason() const noexcept { return reason_; }

    const char* what() const noexcept override { return msg_.c_str(); }

private:
    Reason      reason_;
    std::string msg_;
};

}  // namespace slabkit

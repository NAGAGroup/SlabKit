#pragma once

// =============================================================================
// naga/slabkit/slabkit.hpp — umbrella
// =============================================================================
//
// slabkit is USM memory for SYCL programs that run on several devices, with
// owning handles and one in-order stream of work:
//
//   context       the device set, one sycl::context per backend, and the honest
//                 answers to what the machine can do
//   queue         one in-order stream over that context, with placement,
//                 concurrent groups, host callables and memory work
//   malloc_*      uninitialized storage, SYCL's names and signatures
//   make_*        constructed objects, std::make_unique's forms
//   DeviceArena   a bump region over one slab
//   DevicePool    a fixed-node free list over one slab
//
// over three USM kinds and six placements, yielding one handle type.
//
// NOTE the name collision: malloc_device and friends exist in sycl:: too, and
// ADL reaches both from a call whose arguments include a sycl::queue. Call
// slabkit's QUALIFIED.

#include "naga/slabkit/allocator.hpp"
#include "naga/slabkit/arena.hpp"
#include "naga/slabkit/capabilities.hpp"
#include "naga/slabkit/context.hpp"
#include "naga/slabkit/handles.hpp"
#include "naga/slabkit/kinds.hpp"
#include "naga/slabkit/make.hpp"
#include "naga/slabkit/make_shared.hpp"
#include "naga/slabkit/malloc.hpp"
#include "naga/slabkit/pool.hpp"
#include "naga/slabkit/queue.hpp"
#include "naga/slabkit/usm_vector.hpp"

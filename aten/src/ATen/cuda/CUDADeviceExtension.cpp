#include <ATen/cuda/CUDADevice.h>
#include <map>
#include <mutex>
#include <cstdint>
#include <cstddef>
#include <iterator>

namespace {

struct NUMARegion {
    uintptr_t end;
};

std::map<uintptr_t, NUMARegion> numa_regions;
std::mutex numa_regions_mutex;

}

namespace extension {

void registerNUMAPointer(void* ptr, size_t bytes) {
    TORCH_CHECK(ptr != nullptr, "ptr must not be null");
    TORCH_CHECK(bytes > 0, "bytes must be positive");

    const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    TORCH_CHECK(bytes <= UINTPTR_MAX - start, "Address range overflow.");

    const uintptr_t end = start + bytes;

    std::lock_guard<std::mutex> lock(numa_regions_mutex);

    auto next = numa_regions.lower_bound(start);
    TORCH_CHECK(next == numa_regions.end() || next->first >= end, "NUMA region overlaps with an existing region.");

    if (next != numa_regions.begin()) {
        auto prev = std::prev(next);
        TORCH_CHECK(prev->second.end <= start, "NUMA region overlaps with an existing region.");
    }

    numa_regions.emplace(start, NUMARegion{end});
}

void unregisterNUMAPointer(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);

    std::lock_guard<std::mutex> lock(numa_regions_mutex);
    numa_regions.erase(start);
}

bool isNUMAPointer(void* ptr) {
    if (ptr == nullptr) {
        return false;
    }

    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

    std::lock_guard<std::mutex> lock(numa_regions_mutex);

    auto it = numa_regions.upper_bound(addr);
    if (it == numa_regions.begin()) {
        return false;
    }

    --it;
    return addr < it->second.end;
}

}


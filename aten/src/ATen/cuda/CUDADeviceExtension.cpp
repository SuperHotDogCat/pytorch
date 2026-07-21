#include <ATen/cuda/CUDADevice.h>
#include <unordered_set>
namespace {

std::unordered_set<void*> numa_pointers;

}

namespace extension {

void registerNUMAPointer(void* ptr) {
    numa_pointers.insert(ptr);
}

void unregisterNUMAPointer(void* ptr) {
    numa_pointers.erase(ptr);
}

bool isNUMAPointer(void* ptr) {
    return numa_pointers.find(ptr) != numa_pointers.end();
}

}

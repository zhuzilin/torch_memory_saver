#include <atomic>
#include <cstddef>
#include <dlfcn.h>

// Only the runtime ABI is needed; no CUDA toolkit headers or GPU compilation.
namespace {
std::atomic<int> allocation_error{0};
std::atomic<int> registration_error{0};
std::atomic<int> registrations{0};
}

extern "C" void tms_test_set_faults(int allocation, int registration) {
    allocation_error = allocation;
    registration_error = registration;
}

extern "C" int tms_test_registrations() {
    return registrations;
}

extern "C" int cudaMallocHost(void** ptr, std::size_t size) {
    if (allocation_error) {
        return allocation_error;
    }
    using Function = int (*)(void**, std::size_t);
    static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "cudaMallocHost"));
    return original(ptr, size);
}

extern "C" int cudaHostRegister(void* ptr, std::size_t size, unsigned int flags) {
    if (registration_error) {
        return registration_error;
    }
    using Function = int (*)(void*, std::size_t, unsigned int);
    static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "cudaHostRegister"));
    int result = original(ptr, size, flags);
    if (result == 0) {
        ++registrations;
    }
    return result;
}

extern "C" int cudaHostUnregister(void* ptr) {
    using Function = int (*)(void*);
    static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "cudaHostUnregister"));
    int result = original(ptr);
    if (result == 0) {
        --registrations;
    }
    return result;
}

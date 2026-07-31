#include <iostream>
#include "api_forwarder.h"
#include "utils.h"
#include "macro.h"

namespace APIForwarder {
    using CudaMallocFunc = cudaError_t (*)(void**, size_t);
    using CudaFreeFunc = cudaError_t (*)(void*);
#if defined(USE_CUDA)
    using CudaGetDriverEntryPointFunc = cudaError_t (*)(
        const char*, void**, unsigned long long, cudaDriverEntryPointQueryResult*);
    using CudaGetDriverEntryPointByVersionFunc = cudaError_t (*)(
        const char*, void**, unsigned int, unsigned long long,
        cudaDriverEntryPointQueryResult*);
#endif

#if defined(USE_ROCM)
    static constexpr const char* MALLOC_NAME = "hipMalloc";
    static constexpr const char* FREE_NAME = "hipFree";
#else
    static constexpr const char* MALLOC_NAME = "cudaMalloc";
    static constexpr const char* FREE_NAME = "cudaFree";
#endif

    static void *check_dlsym(void *value) {
        if (nullptr == value) {
            std::cerr << "[torch_memory_saver.cpp] dlsym failed dlerror=" << dlerror() << std::endl;
            exit(1);
        }
        return value;
    }

    static CudaMallocFunc real_cuda_malloc_ = NULL;
    static CudaFreeFunc real_cuda_free_ = NULL;
#if defined(USE_CUDA)
    static CudaGetDriverEntryPointFunc real_cuda_get_driver_entry_point_ = NULL;
    static CudaGetDriverEntryPointByVersionFunc real_cuda_get_driver_entry_point_by_version_ = NULL;
#endif

    cudaError_t call_real_cuda_malloc(void **ptr, size_t size) {
        if (C10_UNLIKELY(nullptr == real_cuda_malloc_)) {
            real_cuda_malloc_ = (CudaMallocFunc) check_dlsym(dlsym(RTLD_NEXT, MALLOC_NAME));
        }

        cudaError_t ret = real_cuda_malloc_(ptr, size);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] APIForwarder.call_real_cuda_malloc "
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }

    cudaError_t call_real_cuda_free(void *ptr) {
        if (C10_UNLIKELY(nullptr == real_cuda_free_)) {
            real_cuda_free_ = (CudaFreeFunc) check_dlsym(dlsym(RTLD_NEXT, FREE_NAME));
        }

        cudaError_t ret = real_cuda_free_(ptr);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] APIForwarder.call_real_cuda_free "
                  << " ptr=" << ptr << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }

#if defined(USE_CUDA)
    cudaError_t call_real_cuda_get_driver_entry_point(
        const char* symbol,
        void** func_ptr,
        unsigned long long flags,
        cudaDriverEntryPointQueryResult* status) {
        if (C10_UNLIKELY(nullptr == real_cuda_get_driver_entry_point_)) {
            real_cuda_get_driver_entry_point_ = (CudaGetDriverEntryPointFunc) check_dlsym(
                dlsym(RTLD_NEXT, "cudaGetDriverEntryPoint"));
        }
        return real_cuda_get_driver_entry_point_(symbol, func_ptr, flags, status);
    }

    cudaError_t call_real_cuda_get_driver_entry_point_by_version(
        const char* symbol,
        void** func_ptr,
        unsigned int version,
        unsigned long long flags,
        cudaDriverEntryPointQueryResult* status) {
        if (C10_UNLIKELY(nullptr == real_cuda_get_driver_entry_point_by_version_)) {
            real_cuda_get_driver_entry_point_by_version_ =
                (CudaGetDriverEntryPointByVersionFunc) check_dlsym(
                    dlsym(RTLD_NEXT, "cudaGetDriverEntryPointByVersion"));
        }
        return real_cuda_get_driver_entry_point_by_version_(
            symbol, func_ptr, version, flags, status);
    }
#endif
}

#pragma once
#include <dlfcn.h>
#include "macro.h"

namespace APIForwarder {
    cudaError_t call_real_cuda_malloc(void **ptr, size_t size);
    cudaError_t call_real_cuda_free(void *ptr);
#if defined(USE_CUDA)
    cudaError_t call_real_cuda_get_driver_entry_point(
        const char* symbol,
        void** func_ptr,
        unsigned long long flags,
        cudaDriverEntryPointQueryResult* status);
    cudaError_t call_real_cuda_get_driver_entry_point_by_version(
        const char* symbol,
        void** func_ptr,
        unsigned int version,
        unsigned long long flags,
        cudaDriverEntryPointQueryResult* status);
#endif
}

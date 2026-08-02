#include "utils.h"
#include "core.h"
#include "api_forwarder.h"
#include <cstring>
#include <dlfcn.h>
#include <optional>
#include "macro.h"

// ----------------------------------------------- threadlocal configs --------------------------------------------------

class ThreadLocalConfig {
public:
    std::string current_tag_ = "default";

    bool is_interesting_region() {
        if (!is_interesting_region_.has_value()) {
            is_interesting_region_ = get_bool_env_var("TMS_INIT_ENABLE");
        }
        return is_interesting_region_.value();
    }

    void set_interesting_region(bool value) {
        is_interesting_region_ = value;
    }

    bool enable_cpu_backup() {
        if (!enable_cpu_backup_.has_value()) {
            enable_cpu_backup_ = get_bool_env_var("TMS_INIT_ENABLE_CPU_BACKUP");
        }
        return enable_cpu_backup_.value();
    }

    void set_enable_cpu_backup(bool value) {
        enable_cpu_backup_ = value;
    }

    bool enable_disk_backup() {
        if (!enable_disk_backup_.has_value()) {
            enable_disk_backup_ = get_bool_env_var("TMS_INIT_ENABLE_DISK_BACKUP");
        }
        return enable_disk_backup_.value();
    }

    void set_enable_disk_backup(bool value) {
        enable_disk_backup_ = value;
    }

private:
    std::optional<bool> is_interesting_region_;
    std::optional<bool> enable_cpu_backup_;
    std::optional<bool> enable_disk_backup_;
};
static thread_local ThreadLocalConfig thread_local_config;

// ------------------------------------------------- entrypoints :: hook ------------------------------------------------

#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(
            ptr, CUDAUtils::cu_ctx_get_device(), size, thread_local_config.current_tag_,
            thread_local_config.enable_cpu_backup(), thread_local_config.enable_disk_backup());
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

cudaError_t cudaFree(void *ptr) {
    return TorchMemorySaver::instance().free(ptr);
}

#if defined(USE_CUDA)
extern "C" CUresult tms_cuMemCreate(
    CUmemGenericAllocationHandle* handle,
    size_t size,
    const CUmemAllocationProp* prop,
    unsigned long long flags) {
    if (!thread_local_config.is_interesting_region()) {
        return cuMemCreate(handle, size, prop, flags);
    }
    return TorchMemorySaver::instance().vmm_create(
        handle,
        size,
        prop,
        flags,
        thread_local_config.current_tag_,
        thread_local_config.enable_cpu_backup(),
        thread_local_config.enable_disk_backup());
}

extern "C" CUresult tms_cuMemMap(
    CUdeviceptr ptr,
    size_t size,
    size_t offset,
    CUmemGenericAllocationHandle handle,
    unsigned long long flags) {
    return TorchMemorySaver::instance().vmm_map(ptr, size, offset, handle, flags);
}

extern "C" CUresult tms_cuMemSetAccess(
    CUdeviceptr ptr,
    size_t size,
    const CUmemAccessDesc* desc,
    size_t count) {
    return TorchMemorySaver::instance().vmm_set_access(ptr, size, desc, count);
}

extern "C" CUresult tms_cuMemUnmap(CUdeviceptr ptr, size_t size) {
    return TorchMemorySaver::instance().vmm_unmap(ptr, size);
}

extern "C" CUresult tms_cuMemRelease(CUmemGenericAllocationHandle handle) {
    return TorchMemorySaver::instance().vmm_release(handle);
}

extern "C" CUresult tms_cuMemExportToShareableHandle(
    void* shareable_handle,
    CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType handle_type,
    unsigned long long flags) {
    return TorchMemorySaver::instance().vmm_export_to_shareable_handle(
        shareable_handle, handle, handle_type, flags);
}

static bool is_pytorch_driver_api_caller(void* return_address) {
    Dl_info caller_info = {};
    if (dladdr(return_address, &caller_info) == 0 || caller_info.dli_fname == nullptr) {
        return false;
    }
    // Do not hand logical TMS handles to NCCL, DeepEP, or arbitrary CUDA
    // libraries that happen to resolve the same driver APIs. PyTorch's
    // DriverAPI singleton currently lives in libc10_cuda; keep libtorch_cuda
    // for compatibility with versions that place the resolver there.
    return std::strstr(caller_info.dli_fname, "libc10_cuda") != nullptr ||
        std::strstr(caller_info.dli_fname, "libtorch_cuda") != nullptr;
}

static void maybe_hook_vmm_driver_entry_point(
    const char* symbol,
    void** func_ptr,
    void* return_address) {
    if (func_ptr == nullptr || *func_ptr == nullptr) {
        return;
    }
    if (!is_pytorch_driver_api_caller(return_address)) {
        return;
    }
    if (std::strcmp(symbol, "cuMemCreate") == 0) {
        *func_ptr = reinterpret_cast<void*>(&tms_cuMemCreate);
    } else if (std::strcmp(symbol, "cuMemMap") == 0) {
        *func_ptr = reinterpret_cast<void*>(&tms_cuMemMap);
    } else if (std::strcmp(symbol, "cuMemSetAccess") == 0) {
        *func_ptr = reinterpret_cast<void*>(&tms_cuMemSetAccess);
    } else if (std::strcmp(symbol, "cuMemUnmap") == 0) {
        *func_ptr = reinterpret_cast<void*>(&tms_cuMemUnmap);
    } else if (std::strcmp(symbol, "cuMemRelease") == 0) {
        *func_ptr = reinterpret_cast<void*>(&tms_cuMemRelease);
    } else if (std::strcmp(symbol, "cuMemExportToShareableHandle") == 0) {
        *func_ptr = reinterpret_cast<void*>(&tms_cuMemExportToShareableHandle);
    }
}

extern "C" cudaError_t cudaGetDriverEntryPoint(
    const char* symbol,
    void** func_ptr,
    unsigned long long flags,
    cudaDriverEntryPointQueryResult* status) {
    const cudaError_t result = APIForwarder::call_real_cuda_get_driver_entry_point(
        symbol, func_ptr, flags, status);
    if (result == cudaSuccess) {
        maybe_hook_vmm_driver_entry_point(
            symbol, func_ptr, __builtin_return_address(0));
    }
    return result;
}

extern "C" cudaError_t cudaGetDriverEntryPointByVersion(
    const char* symbol,
    void** func_ptr,
    unsigned int version,
    unsigned long long flags,
    cudaDriverEntryPointQueryResult* status) {
    const cudaError_t result =
        APIForwarder::call_real_cuda_get_driver_entry_point_by_version(
            symbol, func_ptr, version, flags, status);
    if (result == cudaSuccess) {
        maybe_hook_vmm_driver_entry_point(
            symbol, func_ptr, __builtin_return_address(0));
    }
    return result;
}
#endif
#endif

#ifdef TMS_HOOK_MODE_TORCH
extern "C" {
void *tms_torch_malloc(ssize_t size, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] entrypoint::tms_torch_malloc "
              << " size=" << size << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    void *ptr;
    CUDA_ERROR_CHECK(TorchMemorySaver::instance().malloc(
        &ptr, CUDAUtils::cu_device_get(device), size, thread_local_config.current_tag_,
        thread_local_config.enable_cpu_backup(), thread_local_config.enable_disk_backup()));
    return ptr;
}

void tms_torch_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] entrypoint::tms_torch_free "
              << " ptr=" << ptr << " ssize=" << ssize << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    CUDA_ERROR_CHECK(TorchMemorySaver::instance().free(ptr));
}
}
#endif

// ------------------------------------------------- entrypoints :: others ------------------------------------------------

extern "C" {
void tms_set_interesting_region(bool is_interesting_region) {
    thread_local_config.set_interesting_region(is_interesting_region);
}

bool tms_get_interesting_region() {
    return thread_local_config.is_interesting_region();
}

void tms_set_current_tag(const char* tag) {
    SIMPLE_CHECK(tag != nullptr, "tag should not be null");
    thread_local_config.current_tag_ = tag;
}

const char* tms_get_current_tag() {
    return thread_local_config.current_tag_.c_str();
}

bool tms_get_enable_cpu_backup() {
    return thread_local_config.enable_cpu_backup();
}

void tms_set_enable_cpu_backup(bool enable_cpu_backup) {
    thread_local_config.set_enable_cpu_backup(enable_cpu_backup);
}

bool tms_get_enable_disk_backup() {
    return thread_local_config.enable_disk_backup();
}

void tms_set_enable_disk_backup(bool enable_disk_backup) {
    thread_local_config.set_enable_disk_backup(enable_disk_backup);
}

void tms_set_disk_backup_dir(const char* dir) {
    SIMPLE_CHECK(dir != nullptr, "disk backup dir should not be null");
    TorchMemorySaver::instance().set_disk_backup_dir(std::string(dir));
}

void set_memory_margin_bytes(uint64_t value) {
    TorchMemorySaver::instance().set_memory_margin_bytes(value);
}

void tms_pause(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().pause(tag_str);
}

void tms_resume(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().resume(tag_str);
}

uint8_t* tms_get_cpu_backup_pointer(const uint8_t* gpu_ptr, uint64_t size) {
    return TorchMemorySaver::instance().get_cpu_backup_pointer(gpu_ptr, size);
}
}

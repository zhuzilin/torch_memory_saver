#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <unordered_map>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "utils.h"
#include "macro.h"
#include "disk_backend.h"

#if TMS_ROCM_LEGACY_CHUNKED
#include "hardware_amd_support.h"
#endif

enum class AllocationState {
    // Memory is mapped and accessible
    ACTIVE,
    // Memory is unmapped and inaccessible
    PAUSED
};

struct AllocationMetadata {
    size_t raw_size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    void* cpu_backup;
    bool enable_disk_backup;
    DiskBackupSlot disk;

#if TMS_ROCM_LEGACY_CHUNKED
    // ROCm 6.x: Chunked allocation workaround
    size_t aligned_size;
    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<size_t> chunk_sizes;
#else
    // CUDA and ROCm 7.0+: Single allocation handle
    size_t allocation_size;
    CUmemGenericAllocationHandle allocHandle;
#endif
};

#if defined(USE_CUDA)
struct VMMAllocationMetadata {
    size_t allocation_size;
    CUmemAllocationProp prop;
    unsigned long long create_flags;
    CUmemGenericAllocationHandle physical_handle;
    CUdeviceptr ptr;
    size_t map_size;
    size_t map_offset;
    unsigned long long map_flags;
    std::vector<CUmemAccessDesc> access_descs;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    std::shared_ptr<uint8_t> cpu_backup;
    size_t cpu_backup_offset;
    bool enable_disk_backup;
    DiskBackupSlot disk;
};
#endif

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(
        void** ptr,
        CUdevice device,
        size_t raw_size,
        const std::string& tag,
        bool enable_cpu_backup,
        bool enable_disk_backup);
    cudaError_t free(void *ptr);

#if defined(USE_CUDA)
    CUresult vmm_create(
        CUmemGenericAllocationHandle* logical_handle,
        size_t size,
        const CUmemAllocationProp* prop,
        unsigned long long flags,
        const std::string& tag,
        bool enable_cpu_backup,
        bool enable_disk_backup);
    CUresult vmm_map(
        CUdeviceptr ptr,
        size_t size,
        size_t offset,
        CUmemGenericAllocationHandle logical_handle,
        unsigned long long flags);
    CUresult vmm_set_access(
        CUdeviceptr ptr,
        size_t size,
        const CUmemAccessDesc* desc,
        size_t count);
    CUresult vmm_unmap(CUdeviceptr ptr, size_t size);
    CUresult vmm_release(CUmemGenericAllocationHandle logical_handle);
    CUresult vmm_export_to_shareable_handle(
        void* shareable_handle,
        CUmemGenericAllocationHandle logical_handle,
        CUmemAllocationHandleType handle_type,
        unsigned long long flags);
#endif

    void pause(const std::string& tag);
    void resume(const std::string& tag);
    void set_memory_margin_bytes(uint64_t value) {
        memory_margin_bytes_.store(value);
    }
    uint8_t* get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size);
    void set_disk_backup_dir(const std::string& dir) {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        disk_backend_.set_dir(dir);
    }

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
#if defined(USE_CUDA)
    std::unordered_map<CUmemGenericAllocationHandle, VMMAllocationMetadata> vmm_allocation_metadata_;
    std::map<CUdeviceptr, CUmemGenericAllocationHandle> vmm_mapping_index_;
    std::atomic<uint64_t> next_vmm_logical_handle_ = 1;
#endif
    std::atomic<uint64_t> memory_margin_bytes_ = 0;

    // Guarded by allocator_metadata_mutex_.
    DiskBackend disk_backend_;
};

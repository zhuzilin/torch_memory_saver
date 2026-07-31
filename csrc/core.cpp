#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

TorchMemorySaver::TorchMemorySaver()
    : disk_backend_(compute_disk_backup_dir_from_env(), compute_disk_chunk_bytes_from_env()) {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(
    void **ptr,
    CUdevice device,
    size_t raw_size,
    const std::string& tag,
    const bool enable_cpu_backup,
    const bool enable_disk_backup) {
    // Enforce here, not only in the Python layer: an assert is stripped under
    // python -O and bypassed by direct C-API / env-var use.
    SIMPLE_CHECK(!(enable_cpu_backup && enable_disk_backup),
                 "cpu_backup and disk_backup are mutually exclusive");
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(!enable_disk_backup, "disk backup is not supported on the ROCm 6.x legacy chunked path");
    return ROCmHIPImplementation::rocm_malloc(
        ptr,
        device,
        raw_size,
        tag,
        enable_cpu_backup,
        allocation_metadata_,
        allocator_metadata_mutex_);

#else
    const size_t allocation_size = CUDAUtils::cu_mem_get_allocation_size(raw_size, device);
    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + allocation_size > free_bytes) {
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since"
                << " memory_margin_bytes=" << memory_margin_bytes
                << " allocation_size=" << allocation_size
                << " free_bytes=" << free_bytes
                << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    CUmemGenericAllocationHandle allocHandle;

    cudaError_t ret = CUDAUtils::cu_mem_create(&allocHandle, allocation_size, device);
    if (ret != cudaSuccess) {
        return ret;
    }

    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, allocation_size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, allocation_size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, allocation_size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{
                raw_size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, nullptr,
                enable_disk_backup, DiskBackupSlot{}, allocation_size, allocHandle}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " raw_size=" << raw_size
              << " allocation_size=" << allocation_size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#else
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    if (metadata.state == AllocationState::ACTIVE) {
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.allocation_size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    }
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.allocation_size));

    if (nullptr != metadata.cpu_backup) {
        CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
        metadata.cpu_backup = nullptr;
    }

    if (metadata.enable_disk_backup) {
        disk_backend_.release(metadata.disk);
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free "
              << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
              << " metadata.allocation_size=" << metadata.allocation_size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

#if defined(USE_CUDA)
CUresult TorchMemorySaver::vmm_create(
    CUmemGenericAllocationHandle* logical_handle,
    size_t size,
    const CUmemAllocationProp* prop,
    unsigned long long flags,
    const std::string& tag,
    const bool enable_cpu_backup,
    const bool enable_disk_backup) {
    SIMPLE_CHECK(prop != nullptr, "cuMemCreate prop should not be null");
    SIMPLE_CHECK(!(enable_cpu_backup && enable_disk_backup),
                 "cpu_backup and disk_backup are mutually exclusive");

    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + size > free_bytes) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }

    CUmemGenericAllocationHandle physical_handle;
    const CUresult result = cuMemCreate(&physical_handle, size, prop, flags);
    if (result != CUDA_SUCCESS) {
        return result;
    }

    // PyTorch keeps the handle returned by cuMemCreate until it later unmaps
    // the corresponding expandable-segment page. pause() has to release and
    // recreate the physical handle in between, so expose a stable logical
    // handle to PyTorch and translate it at every handle-taking VMM call.
    CUmemGenericAllocationHandle candidate;
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        do {
            const uint64_t id = next_vmm_logical_handle_.fetch_add(1);
            candidate = static_cast<CUmemGenericAllocationHandle>((uint64_t{1} << 63) | id);
        } while (vmm_allocation_metadata_.count(candidate) != 0);

        vmm_allocation_metadata_.emplace(
            candidate,
            VMMAllocationMetadata{
                size,
                *prop,
                flags,
                physical_handle,
                0,
                0,
                0,
                0,
                {},
                tag,
                AllocationState::ACTIVE,
                enable_cpu_backup,
                nullptr,
                0,
                enable_disk_backup,
                DiskBackupSlot{},
            });
    }
    *logical_handle = candidate;
    return CUDA_SUCCESS;
}

CUresult TorchMemorySaver::vmm_map(
    CUdeviceptr ptr,
    size_t size,
    size_t offset,
    CUmemGenericAllocationHandle logical_handle,
    unsigned long long flags) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    auto it = vmm_allocation_metadata_.find(logical_handle);
    if (it == vmm_allocation_metadata_.end()) {
        return cuMemMap(ptr, size, offset, logical_handle, flags);
    }

    VMMAllocationMetadata& metadata = it->second;
    SIMPLE_CHECK(metadata.state == AllocationState::ACTIVE,
                 "cannot map a paused VMM allocation");
    SIMPLE_CHECK(metadata.ptr == 0, "VMM allocation is already mapped");
    const CUresult result = cuMemMap(ptr, size, offset, metadata.physical_handle, flags);
    if (result == CUDA_SUCCESS) {
        metadata.ptr = ptr;
        metadata.map_size = size;
        metadata.map_offset = offset;
        metadata.map_flags = flags;
        vmm_mapping_index_.emplace(ptr, logical_handle);
    }
    return result;
}

CUresult TorchMemorySaver::vmm_set_access(
    CUdeviceptr ptr,
    size_t size,
    const CUmemAccessDesc* desc,
    size_t count) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    const CUdeviceptr end = ptr + size;
    auto mapping_it = vmm_mapping_index_.lower_bound(ptr);
    while (mapping_it != vmm_mapping_index_.end() && mapping_it->first < end) {
        auto metadata_it = vmm_allocation_metadata_.find(mapping_it->second);
        SIMPLE_CHECK(metadata_it != vmm_allocation_metadata_.end(),
                     "VMM mapping index points to missing allocation");
        VMMAllocationMetadata& metadata = metadata_it->second;
        if (metadata.ptr < end && ptr < metadata.ptr + metadata.map_size) {
            for (size_t i = 0; i < count; ++i) {
                auto existing = metadata.access_descs.begin();
                while (existing != metadata.access_descs.end() &&
                       !(existing->location.type == desc[i].location.type &&
                         existing->location.id == desc[i].location.id)) {
                    ++existing;
                }
                if (existing == metadata.access_descs.end()) {
                    metadata.access_descs.push_back(desc[i]);
                } else {
                    *existing = desc[i];
                }
            }
        }
        ++mapping_it;
    }
    return cuMemSetAccess(ptr, size, desc, count);
}

CUresult TorchMemorySaver::vmm_unmap(CUdeviceptr ptr, size_t size) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    auto mapping_it = vmm_mapping_index_.find(ptr);
    if (mapping_it == vmm_mapping_index_.end()) {
        return cuMemUnmap(ptr, size);
    }

    auto metadata_it = vmm_allocation_metadata_.find(mapping_it->second);
    SIMPLE_CHECK(metadata_it != vmm_allocation_metadata_.end(),
                 "VMM mapping index points to missing allocation");
    VMMAllocationMetadata& metadata = metadata_it->second;
    SIMPLE_CHECK(metadata.map_size == size, "partial managed VMM unmap is not supported");

    CUresult result = CUDA_SUCCESS;
    if (metadata.state == AllocationState::ACTIVE) {
        result = cuMemUnmap(ptr, size);
    }
    if (result == CUDA_SUCCESS) {
        vmm_mapping_index_.erase(mapping_it);
        metadata.ptr = 0;
        metadata.map_size = 0;
        metadata.access_descs.clear();
    }
    return result;
}

CUresult TorchMemorySaver::vmm_release(CUmemGenericAllocationHandle logical_handle) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    auto it = vmm_allocation_metadata_.find(logical_handle);
    if (it == vmm_allocation_metadata_.end()) {
        return cuMemRelease(logical_handle);
    }

    VMMAllocationMetadata& metadata = it->second;
    CUresult result = CUDA_SUCCESS;
    if (metadata.state == AllocationState::ACTIVE) {
        result = cuMemRelease(metadata.physical_handle);
    }
    if (result != CUDA_SUCCESS) {
        return result;
    }

    if (metadata.ptr != 0) {
        vmm_mapping_index_.erase(metadata.ptr);
    }
    metadata.cpu_backup.reset();
    if (metadata.enable_disk_backup) {
        disk_backend_.release(metadata.disk);
    }
    vmm_allocation_metadata_.erase(it);
    return CUDA_SUCCESS;
}

CUresult TorchMemorySaver::vmm_export_to_shareable_handle(
    void* shareable_handle,
    CUmemGenericAllocationHandle logical_handle,
    CUmemAllocationHandleType handle_type,
    unsigned long long flags) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    auto it = vmm_allocation_metadata_.find(logical_handle);
    if (it == vmm_allocation_metadata_.end()) {
        return cuMemExportToShareableHandle(
            shareable_handle, logical_handle, handle_type, flags);
    }
    SIMPLE_CHECK(it->second.state == AllocationState::ACTIVE,
                 "cannot export a paused VMM allocation");
    return cuMemExportToShareableHandle(
        shareable_handle, it->second.physical_handle, handle_type, flags);
}
#endif

void TorchMemorySaver::pause(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::ACTIVE) {
            std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        if (metadata.enable_cpu_backup) {
            if (nullptr == metadata.cpu_backup) {
                CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.raw_size));
            }
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpu_backup, ptr, metadata.raw_size, cudaMemcpyDeviceToHost));
        } else if (metadata.enable_disk_backup) {
            disk_backend_.offload(ptr, metadata.raw_size, metadata.disk);
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.allocation_size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
                  << " metadata.allocation_size=" << metadata.allocation_size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
#endif

#if defined(USE_CUDA) && !TMS_ROCM_LEGACY_CHUNKED
    // Back up each contiguous mapped range into one contiguous host range.
    // A single PyTorch tensor may span several expandable-segment handles, and
    // get_cpu_backup_pointer() must still be able to return one zero-copy CPU
    // view for that tensor.
    auto mapping_it = vmm_mapping_index_.begin();
    while (mapping_it != vmm_mapping_index_.end()) {
        auto metadata_it = vmm_allocation_metadata_.find(mapping_it->second);
        SIMPLE_CHECK(metadata_it != vmm_allocation_metadata_.end(),
                     "VMM mapping index points to missing allocation");
        VMMAllocationMetadata& first = metadata_it->second;
        if ((!tag.empty() && first.tag != tag) || !first.enable_cpu_backup) {
            ++mapping_it;
            continue;
        }
        SIMPLE_CHECK(first.state == AllocationState::ACTIVE,
                     "cannot pause VMM allocation that is not active");

        const CUdeviceptr backup_begin = first.ptr;
        size_t backup_size = first.map_size;
        auto range_end = std::next(mapping_it);
        while (range_end != vmm_mapping_index_.end()) {
            auto next_metadata_it = vmm_allocation_metadata_.find(range_end->second);
            SIMPLE_CHECK(next_metadata_it != vmm_allocation_metadata_.end(),
                         "VMM mapping index points to missing allocation");
            VMMAllocationMetadata& next = next_metadata_it->second;
            if (range_end->first != backup_begin + backup_size ||
                next.state != AllocationState::ACTIVE ||
                !next.enable_cpu_backup ||
                next.tag != first.tag) {
                break;
            }
            backup_size += next.map_size;
            ++range_end;
        }

        void* raw_backup = nullptr;
        CUDA_ERROR_CHECK(cudaMallocHost(&raw_backup, backup_size));
        auto backup = std::shared_ptr<uint8_t>(
            static_cast<uint8_t*>(raw_backup),
            [](uint8_t* ptr) {
                if (ptr != nullptr) {
                    const cudaError_t result = cudaFreeHost(ptr);
                    if (result != cudaSuccess) {
                        std::cerr << "[torch_memory_saver.cpp] cudaFreeHost failed for VMM backup: "
                                  << cudaGetErrorString(result) << std::endl;
                    }
                }
            });
        CUDA_ERROR_CHECK(cudaMemcpy(
            backup.get(),
            reinterpret_cast<void*>(backup_begin),
            backup_size,
            cudaMemcpyDeviceToHost));

        size_t backup_offset = 0;
        for (auto assign_it = mapping_it; assign_it != range_end; ++assign_it) {
            VMMAllocationMetadata& metadata =
                vmm_allocation_metadata_.at(assign_it->second);
            metadata.cpu_backup = backup;
            metadata.cpu_backup_offset = backup_offset;
            backup_offset += metadata.map_size;
        }
        mapping_it = range_end;
    }

    for (auto& [logical_handle, metadata] : vmm_allocation_metadata_) {
        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }
        SIMPLE_CHECK(metadata.state == AllocationState::ACTIVE,
                     "cannot pause VMM allocation that is not active");
        SIMPLE_CHECK(metadata.ptr != 0 && metadata.map_size != 0,
                     "cannot pause VMM allocation before it is mapped");

        void* ptr = reinterpret_cast<void*>(metadata.ptr);
        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr,
                         "contiguous VMM CPU backup should not be null");
        } else if (metadata.enable_disk_backup) {
            disk_backend_.offload(ptr, metadata.map_size, metadata.disk);
        }

        CURESULT_CHECK(cuMemUnmap(metadata.ptr, metadata.map_size));
        CURESULT_CHECK(cuMemRelease(metadata.physical_handle));
        metadata.physical_handle = 0;
        metadata.state = AllocationState::PAUSED;
    }
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        CUmemGenericAllocationHandle newAllocHandle;
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(
            &newAllocHandle, metadata.allocation_size, metadata.device));

        CURESULT_CHECK(cuMemMap(
            (CUdeviceptr) ptr, metadata.allocation_size, 0, newAllocHandle, 0));

        CUDAUtils::cu_mem_set_access(ptr, metadata.allocation_size, metadata.device);

        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(ptr, metadata.cpu_backup, metadata.raw_size, cudaMemcpyHostToDevice));

            // TODO may provide a flag to choose whether to free immediately
            // (users may want to lazily free to reduce re-alloc time)
            CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
            metadata.cpu_backup = nullptr;
        } else if (metadata.enable_disk_backup) {
            disk_backend_.onload(ptr, metadata.raw_size, metadata.disk);
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
                  << " metadata.allocation_size=" << metadata.allocation_size
                  << " (old)metadata.allocHandle=" << metadata.allocHandle
                  << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }
#endif

#if defined(USE_CUDA) && !TMS_ROCM_LEGACY_CHUNKED
    for (auto& [logical_handle, metadata] : vmm_allocation_metadata_) {
        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }
        SIMPLE_CHECK(metadata.state == AllocationState::PAUSED,
                     "cannot resume VMM allocation that is not paused");
        SIMPLE_CHECK(metadata.ptr != 0 && metadata.map_size != 0,
                     "cannot resume VMM allocation after it was unmapped");

        CUmemGenericAllocationHandle new_handle;
        CURESULT_CHECK(cuMemCreate(
            &new_handle, metadata.allocation_size, &metadata.prop, metadata.create_flags));
        CURESULT_CHECK(cuMemMap(
            metadata.ptr,
            metadata.map_size,
            metadata.map_offset,
            new_handle,
            metadata.map_flags));
        if (metadata.access_descs.empty()) {
            CUmemAccessDesc access_desc = {};
            access_desc.location = metadata.prop.location;
            access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CURESULT_CHECK(cuMemSetAccess(
                metadata.ptr, metadata.map_size, &access_desc, 1));
        } else {
            CURESULT_CHECK(cuMemSetAccess(
                metadata.ptr,
                metadata.map_size,
                metadata.access_descs.data(),
                metadata.access_descs.size()));
        }

        void* ptr = reinterpret_cast<void*>(metadata.ptr);
        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr,
                         "VMM CPU backup should not be null");
            CUDA_ERROR_CHECK(cudaMemcpy(
                ptr,
                metadata.cpu_backup.get() + metadata.cpu_backup_offset,
                metadata.map_size,
                cudaMemcpyHostToDevice));
            metadata.cpu_backup.reset();
            metadata.cpu_backup_offset = 0;
        } else if (metadata.enable_disk_backup) {
            disk_backend_.onload(ptr, metadata.map_size, metadata.disk);
        }

        metadata.physical_handle = new_handle;
        metadata.state = AllocationState::ACTIVE;
    }
#endif
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED
        size_t total_size = metadata.aligned_size;
#else
        size_t total_size = metadata.raw_size;
#endif

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + total_size)) {
            const size_t offset = query_gpu_ptr - ptr;
            // Disk-backed allocations have no CPU-resident copy; callers must use the resumed GPU tensor.
            if (metadata.enable_disk_backup) {
                return nullptr;
            }
            if (metadata.state == AllocationState::ACTIVE) {
                return nullptr;
            } else {
                SIMPLE_CHECK(nullptr != metadata.cpu_backup,
                    "get_cpu_backup_pointer: found paused allocation but cpu_backup does not exist, do you forget to enable cpu backup");
                return (uint8_t*) metadata.cpu_backup + offset;
            }
        }
    }

#if defined(USE_CUDA)
    const CUdeviceptr query_ptr = reinterpret_cast<CUdeviceptr>(query_gpu_ptr);
    auto mapping_it = vmm_mapping_index_.upper_bound(query_ptr);
    if (mapping_it != vmm_mapping_index_.begin()) {
        --mapping_it;
        auto metadata_it = vmm_allocation_metadata_.find(mapping_it->second);
        SIMPLE_CHECK(metadata_it != vmm_allocation_metadata_.end(),
                     "VMM mapping index points to missing allocation");
        VMMAllocationMetadata& metadata = metadata_it->second;
        if (metadata.ptr <= query_ptr &&
            query_ptr + query_size <= metadata.ptr + metadata.map_size) {
            if (metadata.enable_disk_backup || metadata.state == AllocationState::ACTIVE) {
                return nullptr;
            }
            SIMPLE_CHECK(metadata.cpu_backup != nullptr,
                         "found paused VMM allocation without CPU backup");
            return metadata.cpu_backup.get() + metadata.cpu_backup_offset +
                (query_ptr - metadata.ptr);
        }
        if (metadata.ptr <= query_ptr && query_ptr < metadata.ptr + metadata.map_size &&
            metadata.state == AllocationState::PAUSED && metadata.enable_cpu_backup) {
            const auto backup = metadata.cpu_backup;
            SIMPLE_CHECK(backup != nullptr, "found paused VMM allocation without CPU backup");
            CUdeviceptr covered_end = metadata.ptr + metadata.map_size;
            size_t expected_backup_offset = metadata.cpu_backup_offset + metadata.map_size;
            auto next_mapping = std::next(mapping_it);
            while (covered_end < query_ptr + query_size &&
                   next_mapping != vmm_mapping_index_.end() &&
                   next_mapping->first == covered_end) {
                VMMAllocationMetadata& next =
                    vmm_allocation_metadata_.at(next_mapping->second);
                if (next.cpu_backup != backup ||
                    next.cpu_backup_offset != expected_backup_offset) {
                    break;
                }
                covered_end += next.map_size;
                expected_backup_offset += next.map_size;
                ++next_mapping;
            }
            if (query_ptr + query_size <= covered_end) {
                return backup.get() + metadata.cpu_backup_offset +
                    (query_ptr - metadata.ptr);
            }
        }
    }
#endif

    std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer fail to find backup "
              << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
              << std::endl;
    exit(1);
}

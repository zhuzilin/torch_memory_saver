import gc
import ctypes

import torch
from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    assert hook_mode == "preload"
    torch.cuda.set_device(0)
    torch_memory_saver.memory_margin_bytes = 0

    # A CUDA library such as NCCL may resolve the same VMM entry points. Only
    # PyTorch's driver API resolver should receive the TMS wrappers.
    process_api = ctypes.CDLL(None)
    get_driver_entry_point = process_api.cudaGetDriverEntryPointByVersion
    get_driver_entry_point.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_ulonglong,
        ctypes.POINTER(ctypes.c_int),
    ]
    get_driver_entry_point.restype = ctypes.c_int
    resolved = ctypes.c_void_p()
    status = ctypes.c_int(-1)
    assert get_driver_entry_point(
        b"cuMemCreate",
        ctypes.byref(resolved),
        12000,
        0,
        ctypes.byref(status),
    ) == 0
    assert status.value == 0
    wrapper_address = ctypes.cast(
        process_api.tms_cuMemCreate,
        ctypes.c_void_p,
    ).value
    assert resolved.value != wrapper_address

    model_tensor = torch.arange(
        64 * 1024 * 1024,
        dtype=torch.int32,
        device="cuda",
    )
    sample_indices = [0, 7_654_321, model_tensor.numel() - 1]
    expected = model_tensor[sample_indices].cpu()
    model_ptr = model_tensor.data_ptr()
    torch.cuda.synchronize()

    snapshot = torch.cuda.memory_snapshot()
    assert any(segment["is_expandable"] for segment in snapshot)
    free_before_pause, _ = torch.cuda.mem_get_info()

    torch_memory_saver.pause()
    free_after_pause, _ = torch.cuda.mem_get_info()
    assert free_after_pause - free_before_pause >= 200 * 1024 * 1024

    # This tensor spans multiple 20 MiB VMM handles. Its backup must still be
    # exposed as one contiguous, zero-copy CPU tensor.
    backup = torch_memory_saver.get_cpu_backup(model_tensor, zero_copy=True)
    assert torch.equal(backup[sample_indices], expected)

    # This is the same ordering Slime uses for weight sync. The cached
    # allocator must not reuse an address from the paused expandable segment.
    original_mem_pool = torch.cuda.MemPool

    def reject_mem_pool(*args, **kwargs):
        raise AssertionError("expandable disable() must not create a MemPool")

    torch.cuda.MemPool = reject_mem_pool
    try:
        with torch_memory_saver.disable():
            # Some users of disable(), such as DeepEP setup, create buffers
            # that intentionally outlive the context.
            persistent_buffer = torch.full(
                (32 * 1024 * 1024,),
                11,
                dtype=torch.uint8,
                device="cuda",
            )
            update_tensor = torch.full(
                (96 * 1024 * 1024,),
                7,
                dtype=torch.uint8,
                device="cuda",
            )
            update_ptr = update_tensor.data_ptr()
            assert update_tensor[123].item() == 7
            assert update_ptr != model_ptr

            snapshot = torch.cuda.memory_snapshot()
            assert any(segment["is_expandable"] for segment in snapshot)
            assert any(not segment["is_expandable"] for segment in snapshot)

            del update_tensor
            gc.collect()
    finally:
        torch.cuda.MemPool = original_mem_pool

    assert persistent_buffer[321].item() == 11
    torch_memory_saver.resume()
    assert torch.equal(model_tensor[sample_indices].cpu(), expected)
    assert persistent_buffer[654].item() == 11

    # disable() restores the expandable allocator for the training hot path.
    post_resume_tensor = torch.empty(
        48 * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    assert any(
        segment["is_expandable"] for segment in torch.cuda.memory_snapshot()
    )
    del post_resume_tensor, persistent_buffer

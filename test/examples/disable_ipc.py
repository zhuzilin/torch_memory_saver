"""Disabled allocations must remain IPC-capable after stream-pool reuse."""
import torch
from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    assert hook_mode == "preload"
    torch_memory_saver.hook_mode = hook_mode
    model = torch.full((1024,), 42, device="cuda")
    torch_memory_saver.pause()
    stream = torch_memory_saver._impl._untracked_streams[torch.cuda.current_device()]

    # Exhaust PyTorch's shared pool. A pooled "untracked" stream would be
    # returned again to NCCL or another user while TMS tracking is enabled.
    pooled_streams = [torch.cuda.Stream() for _ in range(64)]
    for pooled in pooled_streams:
        if pooled.cuda_stream == stream.cuda_stream:
            with torch.cuda.stream(pooled):
                temporary = torch.ones(16 * 1024 * 1024, device="cuda")
                torch.cuda.synchronize()
                del temporary
            break

    with torch_memory_saver.disable():
        update = torch.full((16 * 1024 * 1024,), 7, device="cuda")
        storage = update.untyped_storage()
        shared = storage._share_cuda_()
        storage._release_ipc_counter_cuda(shared[4], shared[5])
        assert update[123].item() == 7

    torch_memory_saver.resume()
    assert model[0].item() == 42
    assert update[456].item() == 7

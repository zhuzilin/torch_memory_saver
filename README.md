# Torch Memory Saver

A PyTorch library that allows tensor memory to be temporarily released and resumed later.

Please refer to https://github.com/sgl-project/sglang/issues/2542#issuecomment-2563641647 for details.

## Examples and Features

### Basic Example

```python
# 1. For tensors that wants to be paused, create them within `region`
with torch_memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. After `pause`, CUDA memory is released for those tensors.
# For example, check `nvidia-smi`'s memory usage to verify.
torch_memory_saver.pause()

# 3. After `resume`, CUDA memory is re-occupied for those tensors.
torch_memory_saver.resume()
```

During the pause, physical memory is released and virtual address is preserved. When resume, virtual address is kept unchanged, while physical memory is re-allocated

### Multiple Tags

Please refer to https://github.com/sgl-project/sglang/issues/7009 for details.

```python
# 1. Create tensors with different tags
with torch_memory_saver.region(tag="type1"):
    tensor1 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

with torch_memory_saver.region(tag="type2"):
    tensor2 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. Pause and resume with different tags selectively
torch_memory_saver.pause("type1")
torch_memory_saver.pause("type2")

torch_memory_saver.resume("type2")
torch_memory_saver.resume("type1")

torch_memory_saver.pause("type1")
torch_memory_saver.resume("type1")
```

### Release Memory in CUDA Graph

Not only does torch_memory_saver make tensors compatible with CUDA graph, but we can also release the memory held by CUDA graph (i.e. the intermediate tensors).

API: Change `torch.cuda.graph(...)` to `torch_memory_saver.cuda_graph(...)`

### CPU Backup

By default, in order to save time, the content is thrown away. This is useful for, for example, KV cache that are to be staled, or model weights that are to be updated.

If you want the tensor content to be kept unchanged, use `enable_cpu_backup`.

```python
with torch_memory_saver.region(enable_cpu_backup=True):
    tensor1 = torch.full((5_000_000_000,), 42, dtype=torch.uint8, device='cuda')

torch_memory_saver.pause()
torch_memory_saver.resume()

assert tensor1[0] == 42, "content is kept unchanged"
```

### Hook Modes

There are two hook modes:

* **preload**: Use `LD_PRELOAD` to hook CUDA's malloc and free API to change allocation behavior.
* **torch**: Use torch's custom allocator API to change allocation behavior.

The mode can be chosen by:

```python
torch_memory_saver.hook_mode = "torch"
```

### Expandable Segments Implementation

The CUDA `preload` hook supports PyTorch's native caching allocator with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The implementation keeps
the caching allocator enabled and works at the CUDA VMM layer:

* The preload library intercepts `cudaGetDriverEntryPoint` and
  `cudaGetDriverEntryPointByVersion`, then replaces the VMM entry points
  resolved by PyTorch (`cuMemCreate`, `cuMemMap`, `cuMemSetAccess`,
  `cuMemUnmap`, `cuMemRelease`, and `cuMemExportToShareableHandle`).
  Resolver calls from NCCL, DeepEP, and unrelated CUDA libraries continue to
  receive the original driver functions.
* PyTorch sees stable logical allocation handles. Torch Memory Saver keeps the
  corresponding physical handles and mapping/access metadata internally.
  `pause()` backs up data when requested, unmaps the original virtual ranges,
  and releases their physical handles. `resume()` creates new physical
  handles, maps them at the same virtual addresses, restores access
  permissions, and copies the data back.
* Adjacent expandable-segment pages share one contiguous pinned-host backup.
  This preserves zero-copy `get_cpu_backup()` for tensors that span several
  CUDA VMM allocation handles and avoids redundant backup allocations.
* `disable()` does not create a `torch.cuda.MemPool` in either expandable or
  legacy mode. Allocations use a dedicated untracked CUDA stream. In expandable
  mode, only the disabled window creates ordinary cached segments; the
  expandable setting is restored afterwards. A device synchronization and
  `empty_cache()` release fully unused temporary segments without disabling
  PyTorch's caching allocator. Live buffers created in the disabled window
  remain valid after the context exits.

Expandable segments require the `preload` hook. The `torch` hook still uses a
custom MemPool for `region()` and therefore rejects this allocator mode.

CUDA VMM remapping changes the underlying physical memory. Communication
libraries that register PyTorch tensor addresses for RDMA or IPC must
deregister those buffers before `pause()` and register them again after
`resume()`.

### Example of RL with CUDA Graph

Please refer to `rl_example.py` for details.

## Development

```bash
make reinstall
```

You can use this command for local testing:

```bash
pytest /path/to/torch_memory_saver/test
```

Or this one to test a single case (e.g. the `simple` one here):

```bash
pytest /path/to/torch_memory_saver/test/test_examples.py::test_simple -s
```

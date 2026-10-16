"""Exercise allocation failures without exhausting the machine's pinned RAM."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from torch_memory_saver.utils import get_binary_path_from_package

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="Host registration requires CUDA",
)


@pytest.fixture(scope="module")
def allocation_faults(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for CUDA allocation fault injection")
    library = tmp_path_factory.mktemp("host-allocation") / "faults.so"
    source = Path(__file__).parent / "native" / "cpu_backup_allocation_faults.cpp"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(source),
            "-ldl",
            "-o",
            str(library),
        ],
        check=True,
    )
    return library


_CHECK_BACKUP = r"""
import ctypes
import gc
import sys
import torch
from torch_memory_saver import torch_memory_saver as saver

mode, allocation_error, registration_error = sys.argv[1:]
saver.hook_mode = "torch" if mode == "torch" else "preload"
saver.memory_margin_bytes = 0
faults = ctypes.CDLL(None)
faults.tms_test_set_faults.argtypes = [ctypes.c_int, ctypes.c_int]
faults.tms_test_registrations.restype = ctypes.c_int

with saver.region(tag="backup", enable_cpu_backup=True):
    tensor = torch.arange(16 * 1024 * 1024, device="cuda", dtype=torch.int32)
expected = tensor.cpu()
original_ptr = tensor.data_ptr()
if mode == "expandable":
    assert any(segment["is_expandable"] for segment in torch.cuda.memory_snapshot())

for cycle in range(3):
    torch.cuda.synchronize()
    faults.tms_test_set_faults(int(allocation_error), int(registration_error))
    saver.pause("backup")
    faults.tms_test_set_faults(0, 0)
    assert faults.tms_test_registrations() > 0
    backup = saver.get_cpu_backup(tensor, zero_copy=True)
    assert torch.equal(backup, expected)
    del backup
    saver.resume("backup")
    assert faults.tms_test_registrations() == 0
    assert tensor.data_ptr() == original_ptr
    assert torch.equal(tensor.cpu(), expected)
    tensor.add_(1)
    expected.add_(1)

# Freeing a paused allocation must unregister its backing storage as well.
faults.tms_test_set_faults(int(allocation_error), 0)
saver.pause("backup")
faults.tms_test_set_faults(0, 0)
del tensor
gc.collect()
# Both region hook modes retain cached blocks in their MemPools. Destroy the
# pools while the allocator's region configuration is still active.
with saver._impl._with_region_config(tag="backup", enable_cpu_backup=True):
    saver._impl._mem_pools.clear()
    torch.cuda.empty_cache()
assert faults.tms_test_registrations() == 0
print("Host registration fallback preserves backup data and releases storage")
"""


def run_backup(allocation_faults, mode, allocation_error, registration_error=0):
    env = os.environ.copy()
    preloads = [str(allocation_faults)]
    if mode != "torch":
        preloads.append(
            str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))
        )
    if env.get("LD_PRELOAD"):
        preloads.append(env["LD_PRELOAD"])
    env["LD_PRELOAD"] = ":".join(preloads)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env["PYTORCH_ALLOC_CONF"] = f"expandable_segments:{mode == 'expandable'}"
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _CHECK_BACKUP,
            mode,
            str(allocation_error),
            str(registration_error),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("mode", ["preload", "torch", "expandable"])
@pytest.mark.parametrize(
    "allocation_error", [1, 2]
)  # Invalid value / allocation failure.
def test_cpu_backup_retries_host_allocation(allocation_faults, mode, allocation_error):
    result = run_backup(allocation_faults, mode, allocation_error)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "retrying with registered host memory" in result.stderr


def test_cpu_backup_propagates_other_cuda_errors(allocation_faults):
    result = run_backup(allocation_faults, "preload", 700)  # Illegal address.
    assert result.returncode != 0
    assert "retrying with registered host memory" not in result.stderr
    assert "illegal memory access" in result.stderr


def test_cpu_backup_propagates_registration_failure(allocation_faults):
    result = run_backup(allocation_faults, "preload", 1, registration_error=2)
    assert result.returncode != 0
    assert "retrying with registered host memory" in result.stderr
    assert "out of memory" in result.stderr

"""
benchmark.py — Demo-grade hardware benchmarking for ViT models.

Protocol:
  Warmup:      10 iterations  (+ 1 CUDA init pass before warmup starts)
  Measurement: 50 iterations
  Batch size:  32
  Energy:      Hardware counter (nvmlDeviceGetTotalEnergyConsumption) when available,
               power-sampling fallback otherwise.
"""

import time
import gc
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    log.warning("pynvml not found — energy measurement disabled. "
                "Fix with: pip install nvidia-ml-py3")


BASELINE_PARAM_COUNT = 85_806_346

DEMO_WARMUP     = 10
DEMO_ITERATIONS = 50
DEMO_BATCH_SIZE = 32

# Progress callback type:  cb(phase, current, total, message)
ProgressCB = Optional[Callable[[str, int, int, str], None]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_batch(batch_size: int, device: str) -> torch.Tensor:
    return torch.randn(batch_size, 3, 224, 224, device=device)


def _cuda_init(model: nn.Module, device: str):
    """
    Single throwaway forward pass (batch=1) to trigger CUDA JIT / cuDNN
    autotuning before any timed measurement.  Prevents the first warmup
    iteration from absorbing the full CUDA init cost (~seconds).
    """
    if device != 'cuda':
        return
    dummy = _make_dummy_batch(1, device)
    with torch.no_grad():
        _ = model(pixel_values=dummy)
    torch.cuda.synchronize()
    del dummy


def _warmup(model: nn.Module, pixel_values: torch.Tensor,
            n: int, device: str, cb: ProgressCB = None):
    model.eval()
    with torch.no_grad():
        for i in range(n):
            _ = model(pixel_values=pixel_values)
            if device == 'cuda':
                torch.cuda.synchronize()
            if cb:
                cb('warmup', i + 1, n, f'Warming up… {i + 1}/{n}')


def _check_nvml_energy_counter():
    """Returns (supported: bool, reason: str). Fully self-contained NVML init/shutdown."""
    if not NVML_AVAILABLE:
        return False, 'pynvml not installed (pip install nvidia-ml-py3)'
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        try:
            val = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            pynvml.nvmlShutdown()
            return True, f'hw_counter OK on {gpu_name} (test={val} mJ)'
        except pynvml.NVMLError as e:
            pynvml.nvmlShutdown()
            return False, f'hw_counter not supported on {gpu_name}: {e}'
    except Exception as e:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return False, f'NVML init failed: {e}'


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def measure_latency(
    model: nn.Module,
    device: str,
    num_warmup: int = DEMO_WARMUP,
    num_iterations: int = DEMO_ITERATIONS,
    batch_size: int = DEMO_BATCH_SIZE,
    cb: ProgressCB = None,
) -> Dict[str, Any]:
    model.eval()
    model.to(device)

    if cb:
        cb('init', 0, 1, 'CUDA init pass…')
    _cuda_init(model, device)
    if cb:
        cb('init', 1, 1, 'CUDA ready')

    pixel_values = _make_dummy_batch(batch_size, device)
    _warmup(model, pixel_values, num_warmup, device, cb)

    if cb:
        cb('latency', 0, num_iterations, 'Starting latency measurement…')

    latencies_ms = []
    with torch.no_grad():
        for i in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(pixel_values=pixel_values)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            if cb:
                cb('latency', i + 1, num_iterations,
                   f'Latency iter {i + 1}/{num_iterations} — {latencies_ms[-1]:.1f} ms')

    mean_batch_ms  = float(np.mean(latencies_ms))
    std_batch_ms   = float(np.std(latencies_ms))
    mean_sample_ms = mean_batch_ms / batch_size

    log.info(f'Latency: {mean_batch_ms:.2f} ± {std_batch_ms:.2f} ms/batch on {device}')

    return {
        'mean_batch_latency_ms':  mean_batch_ms,
        'std_batch_latency_ms':   std_batch_ms,
        'mean_sample_latency_ms': mean_sample_ms,
        'batch_size':             batch_size,
        'num_iterations':         num_iterations,
        'device':                 device,
    }


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def _get_background_power(handle, n_samples: int = 20, interval_s: float = 0.05) -> float:
    powers = []
    for _ in range(n_samples):
        powers.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        time.sleep(interval_s)
    return float(np.median(powers))


def measure_energy(
    model: nn.Module,
    device: str,
    num_warmup: int = DEMO_WARMUP,
    num_iterations: int = DEMO_ITERATIONS,
    batch_size: int = DEMO_BATCH_SIZE,
    cb: ProgressCB = None,
) -> Dict[str, Any]:
    empty = {
        'avg_energy_per_sample_mj': None,
        'std_energy_per_sample_mj': None,
        'background_power_w':       None,
        'method':                   None,
        'supported':                False,
        'nvml_reason':              None,
    }

    if not NVML_AVAILABLE:
        empty['method'] = 'unavailable'
        empty['nvml_reason'] = 'pynvml not installed (pip install nvidia-ml-py3)'
        return empty

    if device != 'cuda':
        empty['method'] = 'unavailable'
        empty['nvml_reason'] = 'device is not cuda'
        return empty

    hw_supported, hw_reason = _check_nvml_energy_counter()
    log.info(f'NVML check: {hw_reason}')

    if cb:
        cb('energy_init', 0, 1, 'Measuring background power…')

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    background_power_w = _get_background_power(handle)

    if cb:
        cb('energy_init', 1, 1, f'Background power: {background_power_w:.1f} W')

    model.eval()
    model.to(device)
    pixel_values = _make_dummy_batch(batch_size, device)
    _warmup(model, pixel_values, num_warmup, device, cb)

    gc.disable()
    energy_per_sample_list = []

    try:
        if hw_supported:
            # Method A: hardware energy counter (most accurate)
            if cb:
                cb('energy', 0, num_iterations, 'Energy measurement (hw counter)…')
            with torch.no_grad():
                for i in range(num_iterations):
                    torch.cuda.synchronize()
                    e_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    t0 = time.perf_counter()
                    _ = model(pixel_values=pixel_values)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    e_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

                    energy_consumed_mj = e_after - e_before
                    bg_mj = background_power_w * (t1 - t0) * 1000.0
                    net_mj = max(0.0, energy_consumed_mj - bg_mj)
                    energy_per_sample_list.append(net_mj / batch_size)

                    if cb:
                        cb('energy', i + 1, num_iterations,
                           f'Energy iter {i + 1}/{num_iterations} — '
                           f'{energy_per_sample_list[-1]:.4f} mJ/sample')
            method = 'hw_counter'

        else:
            # Method B: power sampling fallback
            if cb:
                cb('energy', 0, num_iterations,
                   f'Energy measurement (power sampling — {hw_reason})…')
            with torch.no_grad():
                for i in range(num_iterations):
                    torch.cuda.synchronize()
                    p_before = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    t0 = time.perf_counter()
                    _ = model(pixel_values=pixel_values)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    p_after = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

                    avg_power_w = (p_before + p_after) / 2.0
                    net_power_w = max(0.0, avg_power_w - background_power_w)
                    net_mj = (net_power_w * (t1 - t0) * 1000.0) / batch_size
                    energy_per_sample_list.append(net_mj)

                    if cb:
                        cb('energy', i + 1, num_iterations,
                           f'Energy iter {i + 1}/{num_iterations} — '
                           f'{energy_per_sample_list[-1]:.4f} mJ/sample')
            method = 'power_sampling'

    finally:
        gc.enable()
        gc.collect()
        pynvml.nvmlShutdown()

    arr = np.array(energy_per_sample_list)
    return {
        'avg_energy_per_sample_mj': float(np.mean(arr)),
        'std_energy_per_sample_mj': float(np.std(arr)),
        'background_power_w':       background_power_w,
        'method':                   method,
        'supported':                True,
        'nvml_reason':              hw_reason,
        'num_samples':              len(arr),
    }


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Full benchmark for one model
# ---------------------------------------------------------------------------

def benchmark_model(
    model: nn.Module,
    model_name: str,
    device: str,
    baseline_latency_ms: Optional[float] = None,
    baseline_energy_mj:  Optional[float] = None,
    baseline_params:     int = BASELINE_PARAM_COUNT,
    cb: ProgressCB = None,
) -> Dict[str, Any]:
    log.info(f'Benchmarking {model_name} on {device}')
    total_params = count_params(model)

    latency_res = measure_latency(model, device, cb=cb)
    energy_res  = measure_energy(model, device, cb=cb)

    mean_lat = latency_res['mean_batch_latency_ms']
    avg_e    = energy_res['avg_energy_per_sample_mj']

    speedup = (baseline_latency_ms / mean_lat) if baseline_latency_ms else None
    energy_reduction = (
        (1.0 - avg_e / baseline_energy_mj) * 100.0
        if (baseline_energy_mj and avg_e is not None)
        else None
    )
    param_reduction = (1.0 - total_params / baseline_params) * 100.0

    return {
        'model_name':               model_name,
        'device':                   device,
        'mean_batch_latency_ms':    mean_lat,
        'std_batch_latency_ms':     latency_res['std_batch_latency_ms'],
        'mean_sample_latency_ms':   latency_res['mean_sample_latency_ms'],
        'speedup':                  speedup,
        'avg_energy_per_sample_mj': avg_e,
        'std_energy_per_sample_mj': energy_res['std_energy_per_sample_mj'],
        'energy_reduction_pct':     energy_reduction,
        'energy_method':            energy_res['method'],
        'energy_supported':         energy_res['supported'],
        'nvml_reason':              energy_res.get('nvml_reason'),
        'total_params':             total_params,
        'param_reduction_pct':      param_reduction,
        'batch_size':               DEMO_BATCH_SIZE,
        'warmup_iters':             DEMO_WARMUP,
        'measure_iters':            DEMO_ITERATIONS,
    }

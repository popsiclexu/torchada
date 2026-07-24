# Adapted from SGLang's tuning_fused_moe_triton_sep.py, which is based on
# https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from tqdm import tqdm

from torchada.triton.autotune.fused_moe.tune_moe import (
    MoeRunnerConfig,
    _resolve_dtype_str,
    _resolve_torch_dtype,
)
from torchada.triton.autotune.fused_moe.utils import (
    BenchmarkConfig,
    get_config_filename,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    sort_config,
)
from torchada.triton.kernels.moe.kernel import invoke_fused_moe_kernel
from torchada.triton.runtime.fused_moe.config import (
    get_config_dtype_str,
    get_config_file_name,
    override_config,
)
from torchada.triton.runtime.fused_moe.fused_moe import moe_align_block_size
from torchada.triton.runtime.fused_moe.router import TopKConfig, select_experts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    """Fallback activation for environments without the sgl_kernel wheel."""
    split = x.shape[-1] // 2
    out.copy_(torch.nn.functional.silu(x[..., :split]) * x[..., split:])


_is_hip = False
_TOPK_IDS_CACHE: Dict[str, List[torch.Tensor]] = {}


def _load_topk_id_samples(topk_ids_dir: str) -> List[torch.Tensor]:
    """Load captured top-k samples once per worker process.

    Older versions hard-coded DeepSeek-V3's layer3..60 and 58 routed layers.
    DeepSeek-V2 has first_k_dense_replace=1 and 59 routed layers, so discover
    the files written by the capture hook instead of encoding a model layout.
    """
    if topk_ids_dir in _TOPK_IDS_CACHE:
        return _TOPK_IDS_CACHE[topk_ids_dir]

    paths = sorted(glob.glob(os.path.join(topk_ids_dir, "topk_ids_layer*_idx*.pt")))
    if not paths:
        raise FileNotFoundError(
            f"No captured top-k files found under {topk_ids_dir!r}; "
            "run a graph-disabled capture first"
        )

    samples: List[torch.Tensor] = []
    for path in paths:
        try:
            sample = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            sample = torch.load(path, map_location="cpu")
        if not isinstance(sample, torch.Tensor) or sample.ndim != 2:
            raise ValueError(f"Invalid top-k sample {path}: expected rank-2 tensor")
        samples.append(sample.contiguous())

    _TOPK_IDS_CACHE[topk_ids_dir] = samples
    logger.info("Loaded %d captured top-k samples from %s", len(samples), topk_ids_dir)
    return samples


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    topk_ids_dir: str,
    block_shape: List[int] = None,
    num_iters: int = 100,
) -> float:
    ncu_enable = os.getenv("NCU_ENABLE", "0") == "1"
    if ncu_enable:
        num_iters = 1
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    else:
        w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype)
        w2 = torch.randn(num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype)
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
    captured_topk_ids = _load_topk_id_samples(topk_ids_dir)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size), dtype=torch.float32)
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(num_experts, shard_intermediate_size, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand((num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
            w2_scale = torch.rand((num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32)

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_config = TopKConfig(
        top_k=topk,
        renormalize=True,
    )
    topk_output = select_experts(hidden_states, input_gating, topk_config)

    def prepare(i: int):
        input_gating = gating_output[i]
        topk_ids = captured_topk_ids[i % len(captured_topk_ids)]
        new_topk_output = select_experts(hidden_states, input_gating, topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        tokens, _topk = topk_output.topk_ids.shape
        if topk_ids.shape[0] < tokens or topk_ids.shape[1] < _topk:
            raise ValueError(
                f"Captured top-k shape {tuple(topk_ids.shape)} is smaller than "
                f"requested {(tokens, _topk)}"
            )
        topk_output.topk_ids.copy_(topk_ids[:tokens, :_topk])
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def benchmark_graph_variant(use_tma: bool) -> Tuple[float, float]:
        """Time the two MoE GEMMs through graph replay on one real route sample.

        The graph captures ten identical kernel launches, matching SGLang's
        existing fused-MoE tuning methodology while removing eager Python
        launch overhead. The chosen sample is the median captured route by
        unique-expert count so tuning is not biased by an extreme layer.
        """
        ranked_samples = sorted(
            captured_topk_ids,
            key=lambda sample: int(torch.unique(sample).numel()),
        )
        graph_sample = ranked_samples[len(ranked_samples) // 2]
        new_topk_output = select_experts(hidden_states, gating_output[0], topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        tokens, sample_topk = topk_output.topk_ids.shape
        if graph_sample.shape[0] < tokens or graph_sample.shape[1] < sample_topk:
            raise ValueError(
                f"Captured top-k shape {tuple(graph_sample.shape)} is smaller than "
                f"requested {(tokens, sample_topk)}"
            )
        topk_output.topk_ids.copy_(graph_sample[:tokens, :sample_topk])
        topk_output.router_logits.copy_(new_topk_output.router_logits)

        moe_runner_config = MoeRunnerConfig(inplace=True)
        topk_weights, topk_ids, _ = topk_output
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], num_experts
        )
        M = hidden_states.shape[0]
        E, N, _ = w1.shape
        padded_tokens = min(M * sample_topk, E + 1) * (config["BLOCK_SIZE_M"] - 1) if use_tma else 0
        total_tokens = M * sample_topk + padded_tokens
        cache = torch.empty(
            total_tokens * max(N, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache1 = cache[: total_tokens * N].view(total_tokens, N)
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache3 = cache[: M * sample_topk * w2.shape[1]].view(
            M, sample_topk, w2.shape[1]
        )
        compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        apply_router_weight_on_input = moe_runner_config.apply_router_weight_on_input

        def kernel0() -> None:
            with override_config(config):
                invoke_fused_moe_kernel(
                    hidden_states,
                    w1,
                    None,
                    intermediate_cache1,
                    None,
                    w1_scale,
                    None,
                    topk_weights,
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    apply_router_weight_on_input,
                    sample_topk,
                    config,
                    compute_type=compute_type,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=block_shape,
                    b_use_tma=use_tma,
                    c_sorted=use_tma,
                    filter_expert=False,
                )

        def kernel1() -> None:
            with override_config(config):
                invoke_fused_moe_kernel(
                    intermediate_cache2,
                    w2,
                    None,
                    intermediate_cache3,
                    a2_scale,
                    w2_scale,
                    None,
                    topk_weights,
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    not apply_router_weight_on_input,
                    1,
                    config,
                    compute_type=compute_type,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=block_shape,
                    a_use_tma=use_tma,
                    b_use_tma=use_tma,
                    filter_expert=False,
                )

        # Compile before capture and materialize the activation consumed by
        # the separately captured down-projection graph.
        kernel0()
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        kernel1()
        torch.cuda.synchronize()

        def capture_and_time(fn) -> float:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    fn()
            torch.cuda.synchronize()
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()

            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
            for i in range(num_iters):
                start_events[i].record()
                graph.replay()
                end_events[i].record()
            torch.cuda.synchronize()
            latency_us = (
                sum(start_events[i].elapsed_time(end_events[i]) for i in range(num_iters))
                / (num_iters * 10)
                * 1000
            )
            graph.reset()
            return latency_us

        return capture_and_time(kernel0), capture_and_time(kernel1)

    if os.getenv("TORCHADA_TUNE_USE_GRAPH", "1") == "1":
        no_tma0, no_tma1 = benchmark_graph_variant(False)
        # Some MUSA Triton builds expose TensorDescriptor but do not provide
        # the CUDA-side global allocator API (`triton.set_allocator`) needed
        # to launch TMA descriptors.  Keep graph-aware timing for the
        # supported non-TMA path and mark TMA as unavailable instead of
        # failing every batch-size task.
        if os.getenv("TORCHADA_TUNE_DISABLE_TMA", "1") == "1":
            return no_tma0, no_tma0, no_tma1, no_tma1
        tma0, tma1 = benchmark_graph_variant(True)
        return no_tma0, tma0, no_tma1, tma1

    moe_use_tma = False

    def run():
        moe_runner_config = MoeRunnerConfig(
            inplace=True,
        )
        topk_weights, topk_ids, _ = topk_output

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], num_experts
        )
        M = hidden_states.shape[0]
        E, N, _ = w1.shape

        topk = topk_ids.shape[1]
        padded_tokens = min(M * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1) if moe_use_tma else 0
        total_tokens = M * topk + padded_tokens
        cache = torch.empty(
            total_tokens * max(N, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache1 = cache[: total_tokens * N].view(
            (total_tokens, N),
        )
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache3 = cache[: M * topk * w2.shape[1]].view(
            (M, topk, w2.shape[1]),
        )

        compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        apply_router_weight_on_input = moe_runner_config.apply_router_weight_on_input

        with override_config(config):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(10 if not ncu_enable else 1):
                invoke_fused_moe_kernel(
                    hidden_states,
                    w1,
                    None,
                    intermediate_cache1,
                    None,
                    w1_scale,
                    None,
                    topk_weights,
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    apply_router_weight_on_input,
                    topk_ids.shape[1],
                    config,
                    compute_type=compute_type,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=block_shape,
                    b_use_tma=moe_use_tma,
                    c_sorted=moe_use_tma,
                    filter_expert=False,
                )
            end_event.record()
            end_event.synchronize()
            time_cost0 = start_event.elapsed_time(end_event)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

            silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            for _ in range(10 if not ncu_enable else 1):
                invoke_fused_moe_kernel(
                    intermediate_cache2,
                    w2,
                    None,
                    intermediate_cache3,
                    a2_scale,
                    w2_scale,
                    None,
                    topk_weights,
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    not apply_router_weight_on_input,
                    1,
                    config,
                    compute_type=compute_type,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=block_shape,
                    a_use_tma=moe_use_tma,
                    b_use_tma=moe_use_tma,
                    filter_expert=False,
                )
            end_event.record()
            end_event.synchronize()
            time_cost1 = start_event.elapsed_time(end_event)
        return time_cost0, time_cost1

    # JIT compilation & warmup
    if not ncu_enable:
        moe_use_tma = False
        run()
        moe_use_tma = True
        run()
    latencies: List[float] = []
    latencies1: List[float] = []
    latencies_tma: List[float] = []
    latencies1_tma: List[float] = []

    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()
        moe_use_tma = False
        t0, t1 = run()
        torch.cuda.synchronize()
        latencies.append(t0)
        latencies1.append(t1)

        moe_use_tma = True
        t0, t1 = run()
        torch.cuda.synchronize()
        latencies_tma.append(t0)
        latencies1_tma.append(t1)

    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    avg_tma = sum(latencies_tma) / (num_iters * 10) * 1000  # us
    avg1 = sum(latencies1) / (num_iters * 10) * 1000  # us
    avg1_tma = sum(latencies1_tma) / (num_iters * 10) * 1000  # us

    return avg, avg_tma, avg1, avg1_tma


class BestConfigTrace:
    def __init__(self, name):
        self.name = name
        self.config = None
        self.time_cost = float("inf")
        self.time_cost_all = (
            None  # kernel0 without tma,, kernel0 with tma, kernel1 without tma, kernel1 with tma
        )

    def update(self, config, time_cost, time_cost_all):
        if time_cost < self.time_cost:
            print(
                f"New best config for {self.name}: {config}, {time_cost=}, {time_cost_all=}, org: {self.config}, {self.time_cost_all}",
                flush=True,
            )
            self.config = config
            self.time_cost = time_cost
            self.time_cost_all = time_cost_all

    @property
    def total_time(self):
        return self.time_cost_all[0] + min(self.time_cost_all[2], self.time_cost_all[3])

    def config_dict(self, down_moe=False):
        if not down_moe:
            return self.config
        else:
            # SGLang only consumes USE_TMA when the down-MoE path is enabled.
            # If TMA is not selected, omitting the optional key is important:
            # older MUSA runners pass the down config through unchanged when
            # GEMV dispatch disables TMA, and Triton would reject USE_TMA as
            # an unknown kernel argument even when its value is false.
            use_tma = self.time_cost_all[2] > self.time_cost_all[3]
            return {
                **self.config,
                **({"USE_TMA": True} if use_tma else {}),
            }


class BenchmarkWorker:

    def __init__(self, seed: int, device_id: int = 0) -> None:
        torch.cuda.set_device(device_id)
        torch.set_default_device("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU.
        self.device_id = device_id

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        block_shape: List[int],
        cfg: Dict[str, int],
        topk_ids_dir: str,
    ) -> Tuple[Dict[str, int], float]:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        dtype_str = get_config_dtype_str(
            dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        with torch.cuda.device(self.device_id):
            kernel_time = benchmark_config(
                cfg,
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                topk_ids_dir,
                block_shape,
            )
        return cfg, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        block_shape: List[int],
        search_space: List[Dict[str, int]],
        topk_ids_dir: str,
    ) -> Dict[str, int]:
        trace0 = BestConfigTrace("kernel0")
        trace1 = BestConfigTrace("kernel1")
        trace2 = BestConfigTrace("kernel all")

        with torch.cuda.device(self.device_id):
            for config in tqdm(search_space):
                try:
                    kt0_no_tma, kt0_tma, kt1_no_tma, kt1_tma = benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        topk_ids_dir,
                        block_shape,
                        num_iters=10,
                    )
                except (triton.runtime.autotuner.OutOfResources, RuntimeError, AssertionError):
                    # Some configurations may be invalid and fail to compile.
                    continue
                kt0 = kt0_no_tma
                kt1 = min(kt1_no_tma, kt1_tma)
                trace0.update(
                    config,
                    kt0,
                    (kt0_no_tma, kt0_tma, kt1_no_tma, kt1_tma),
                )
                trace1.update(
                    config,
                    kt1,
                    (kt0_no_tma, kt0_tma, kt1_no_tma, kt1_tma),
                )
                trace2.update(
                    config,
                    kt0 + kt1,
                    (kt0_no_tma, kt0_tma, kt1_no_tma, kt1_tma),
                )

        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert trace0.config is not None
        assert trace1.config is not None
        print(
            f"{num_tokens=}, {trace0.config=}, {trace0.time_cost_all=}, {trace1.config=}, {trace1.time_cost_all=}"
        )
        if trace0.config["BLOCK_SIZE_M"] != trace1.config["BLOCK_SIZE_M"]:
            best_trace = trace0 if trace0.total_time < trace1.total_time else trace1
            best_trace = best_trace if best_trace.total_time < trace2.total_time else trace2
            return (
                best_trace.config_dict(),
                best_trace.config_dict(True),
                best_trace.time_cost_all,
                best_trace.time_cost_all,
            )
        return (
            trace0.config_dict(),
            trace1.config_dict(True),
            trace0.time_cost_all,
            trace1.time_cost_all,
        )


def save_configs_sep(
    configs: Dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: List[int],
    down_moe: bool = False,
) -> None:
    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
    )

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = get_config_file_name(
        num_experts,
        shard_intermediate_size // 2,
        dtype_str,
        block_shape,
        down_moe=down_moe,
    )

    default_config_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.environ.get("SGLANG_MOE_CONFIG_DIR", default_config_dir)
    version_dir = f"triton_{triton.__version__.replace('.', '_')}"
    config_dir = os.path.join(config_dir, "configs", version_dir)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, filename)

    configs_to_write = {str(batch_size): config for batch_size, config in configs.items()}
    if os.getenv("TORCHADA_TUNE_MERGE_EXISTING_CONFIG", "0") == "1":
        base_path = config_path
        if down_moe and not os.path.exists(base_path):
            base_path = config_path.replace("_down.json", ".json")
        if os.path.exists(base_path):
            with open(base_path) as f:
                merged_configs = json.load(f)
            merged_configs.update(configs_to_write)
            configs_to_write = merged_configs
            print(
                f"Merged {len(configs)} tuned entries with base config {base_path}",
                flush=True,
            )

    print(f"Writing best config to {config_path}...")
    with open(config_path, "w") as f:
        json.dump(configs_to_write, f, indent=4)
        f.write("\n")


def _tune_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
) -> None:
    """Run one batch-size tuning task per queue item on a dedicated GPU."""
    worker = BenchmarkWorker(seed, device_id=gpu_id)
    while True:
        task = task_queue.get()
        if task is None:
            break
        task_id, tune_args = task
        try:
            result = worker.tune(*tune_args)
            result_queue.put((task_id, result, None))
        except Exception as exc:
            logger.exception("Tuning failed on GPU %d for task %d", gpu_id, task_id)
            result_queue.put((task_id, None, repr(exc)))
    result_queue.put(None)


def _run_tuning_tasks(
    tune_args_list: List[Tuple], seed: int
) -> List[Tuple[Dict[str, int], Dict[str, int], Tuple, Tuple]]:
    """Distribute batch-size tuning tasks across GPUs with multiprocessing."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices found")

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for task_id, tune_args in enumerate(tune_args_list):
        task_queue.put((task_id, tune_args))
    for _ in range(num_gpus):
        task_queue.put(None)

    workers = []
    for gpu_id in range(num_gpus):
        process = mp.Process(
            target=_tune_worker,
            args=(gpu_id, task_queue, result_queue, seed),
        )
        process.start()
        workers.append(process)

    results: List[Optional[Tuple[Dict[str, int], Dict[str, int], Tuple, Tuple]]] = [None] * len(
        tune_args_list
    )
    errors = []
    active_workers = num_gpus
    with tqdm(total=len(tune_args_list), desc="Tuning batch sizes", unit="task") as pbar:
        while active_workers > 0:
            result = result_queue.get()
            if result is None:
                active_workers -= 1
                continue
            task_id, value, error = result
            if error is not None:
                errors.append(f"task {task_id}: {error}")
            else:
                results[task_id] = value
            pbar.update(1)

    for process in workers:
        process.join()
    if errors:
        raise RuntimeError("\n".join(errors))
    if any(result is None for result in results):
        raise RuntimeError("One or more tuning workers exited without a result")
    return results  # type: ignore[return-value]


def _save_tuning_results(
    results: List[Tuple[Dict[str, int], Dict[str, int], Tuple, Tuple]],
    batch_sizes: List[int],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: List[int],
) -> None:
    by_batch_size = {batch_size: result for batch_size, result in zip(batch_sizes, results)}
    configs0 = {
        batch_size: sort_config(result[0]) for batch_size, result in sorted(by_batch_size.items())
    }
    configs1 = {
        batch_size: sort_config(result[1]) for batch_size, result in sorted(by_batch_size.items())
    }
    save_configs_sep(
        configs0,
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        block_shape,
    )
    save_configs_sep(
        configs1,
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        block_shape,
        down_moe=True,
    )


def _get_search_space(
    block_shape: Optional[List[int]], block_size_m: Optional[List[int]] = None
) -> List[BenchmarkConfig]:
    search_space = get_configs_compute_bound()
    if block_shape is not None:
        block_k = block_shape[1]
        search_space = [config for config in search_space if block_k % config["BLOCK_SIZE_K"] == 0]
    if block_size_m:
        allowed = set(block_size_m)
        search_space = [config for config in search_space if config["BLOCK_SIZE_M"] in allowed]
    return search_space


def main(args: argparse.Namespace):
    if args.use_graph:
        os.environ["TORCHADA_TUNE_USE_GRAPH"] = "1"
    print(args)

    model_config = get_model_config(
        args.model,
        args.tp_size,
        args.ep_size,
        args.disable_shared_experts_fusion,
        args.topk_ids_dir,
    )

    E = model_config["num_experts"]
    # DeepSeek-V2 serving with shared-experts fusion disabled routes six
    # routed experts, while the model config may expose an extra shared
    # expert.  Allow callers to pin the routed top-k to the observed serving
    # value used by the captured workload.
    topk = args.topk if args.topk is not None else model_config["topk"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype_str = _resolve_dtype_str(args.dtype, model_config)
    dtype = _resolve_torch_dtype(dtype_str, model_config)
    block_shape = model_config["block_shape"]

    use_fp8_w8a8 = dtype_str == "fp8_w8a8"
    use_int8_w8a8 = dtype_str == "int8_w8a8"
    use_int8_w8a16 = dtype_str == "int8_w8a16"

    topk_ids_dir = args.topk_ids_dir
    if args.batch_size is None:
        batch_sizes = get_default_batch_sizes()
        batch_sizes.reverse()
    else:
        batch_sizes = sorted({int(value) for item in args.batch_size for value in item.split(",")})
    if len(batch_sizes) == 1:
        if args.tune:
            search_space = _get_search_space(block_shape, args.block_size_m)
            result = _run_tuning_tasks(
                [
                    (
                        batch_sizes[0],
                        E,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        block_shape,
                        search_space,
                        topk_ids_dir,
                    )
                ],
                args.seed,
            )
            _save_tuning_results(
                result,
                batch_sizes,
                E,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                block_shape,
            )
        else:
            worker = BenchmarkWorker(args.seed)
            cfg = {
                "BLOCK_SIZE_M": args.configs[0],
                "BLOCK_SIZE_N": args.configs[1],
                "BLOCK_SIZE_K": args.configs[2],
                "GROUP_SIZE_M": args.configs[3],
                "num_warps": args.configs[4],
                "num_stages": args.configs[5],
            }

            _, (t0, t0_tma, t1, t1_tma) = worker.benchmark(
                batch_sizes[0],
                E,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                block_shape,
                cfg,
                topk_ids_dir,
            )
            print(f"{t0=}, {t0_tma=}, {t1=}, {t1_tma=}")
        return

    assert args.tune
    search_space = _get_search_space(block_shape, args.block_size_m)
    filename = get_config_filename(
        E,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        False,
        False,
        block_shape,
    )
    print(f"Start tuning over {len(search_space)} configurations to create {filename}...")

    start = time.perf_counter()
    tune_args_list = [
        (
            batch_size,
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            block_shape,
            search_space,
            topk_ids_dir,
        )
        for batch_size in batch_sizes
    ]
    configs = _run_tuning_tasks(
        tune_args_list,
        args.seed,
    )
    print(f"{configs=}", flush=True)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(f"tuning_result_{cur_time}.txt", "w") as f:
        print(configs, file=f)
    _save_tuning_results(
        configs,
        batch_sizes,
        E,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        block_shape,
    )
    end = time.perf_counter()
    print(f"Tuning took {end - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Override routed top-k (useful when shared experts are disabled).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=str,
        action="append",
        required=False,
        help="Batch size(s), e.g. --batch-size 1,2,4 or --batch-size 8",
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    parser.add_argument("--configs", type=int, nargs="+", required=False)
    parser.add_argument("--topk-ids-dir", type=str, required=True)
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Rank configs by CUDA/MUSA graph replay rather than eager launch timing.",
    )
    parser.add_argument(
        "--block-size-m",
        type=lambda value: [int(item) for item in value.split(",")],
        default=None,
        help="Optional comma-separated BLOCK_SIZE_M filter, e.g. 16,32.",
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    main(args)

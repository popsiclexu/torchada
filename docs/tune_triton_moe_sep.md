# tune_moe_sep — Separate Fused-MoE Kernel Tuning

`src/torchada/triton/autotune/fused_moe/tune_moe_sep.py` tunes the two Triton
GEMMs in a fused MoE layer independently:

- **up/gate projection** (`kernel0`), which multiplies the hidden states by
  the fused gate/up weights; and
- **down projection** (`kernel1`), which consumes the activated intermediate
  states.

The projections have different matrix shapes and may prefer different tile,
warp, stage, and TMA settings. The tuner therefore writes two configuration
maps: a normal file for `kernel0` and a `_down.json` file for `kernel1`.

This workflow is adapted from SGLang's
[`tuning_fused_moe_triton_sep.py`](https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py),
but has several torchada-specific behaviors:

- captured files are discovered by filename instead of assuming a fixed model
  layer layout;
- batch-size tasks are distributed with Python multiprocessing, without Ray;
- generated files are written under the torchada/SGLang versioned config
  directory; and
- captured expert IDs must already use the local numbering expected by the
  selected EP configuration.

Use the standard [`tune_moe.py`](tune_triton_moe.md) workflow when real routing
samples or independent down-projection tuning are not required.

> This is a device-specific benchmark. Run it on the same CUDA/MUSA device
> type, Triton version, TP/EP topology, dtype, quantization mode, and
> shared-expert setting as the deployment that will consume the result.

## Supported parallel modes

The tool supports TP and EP, including a combination of the two:

| Mode | Example | Meaning |
| --- | --- | --- |
| TP only | `--tp-size 4 --ep-size 1` | Experts are replicated; the intermediate dimension is TP-sharded. |
| EP only | `--tp-size 4 --ep-size 4` | Experts are distributed; there is no additional MoE tensor parallelism inside each EP shard. |
| TP + EP | `--tp-size 8 --ep-size 4` | Experts are distributed across four EP ranks and each local expert is sharded with MoE TP size `8 / 4 = 2`. |

`tp_size` must be divisible by `ep_size`. The capture server and tuner must use
the same values. Changing either value changes the local expert count,
intermediate shape, and possibly the expert-ID numbering.

Model configuration loading covers the same main families as the standard
tuner, including Mixtral, DBRX, Jamba, Qwen MoE/Qwen-VL MoE, DeepSeek V2/V3,
GLM MoE, Llama 4, Grok, Bailing, Nemotron-H, Gemma 4, and compatible custom
architectures exposing the expected MoE fields.

## Requirements

Install torchada in the environment where the benchmark will run:

```bash
pip install -e .
```

The tuning environment must provide:

- PyTorch and Triton with at least one visible CUDA/MUSA device;
- `transformers` and access to the model's `config.json` through a local path,
  Hugging Face, or ModelScope;
- `tqdm`; and
- either `sgl_kernel` or `vllm` for `moe_align_block_size`.

The benchmark generates random hidden states, weights, scales, and router
weights. It loads only the model configuration and the captured expert IDs;
model weights are not required.

The bundled request client additionally requires the OpenAI Python client:

```bash
pip install openai
```

The sep script currently uses its local PyTorch implementation of
SiLU-and-multiply. Installing `sgl_kernel` is still useful when it provides the
selected `moe_align_block_size` implementation, but it does not replace the
activation used by this benchmark.

On MUSA, keep the normal torchada import-order requirement: `import torchada`
must happen before SGLang or another framework imports and caches CUDA APIs.

## Workflow overview

The complete workflow is:

1. add a temporary save hook to the model's MoE routing path;
2. launch the same model with graph capture disabled;
3. send one or more long-prefill requests to produce `topk_ids` samples;
4. verify shape, expert numbering, and TP/EP compatibility;
5. tune one or more active-token counts; and
6. start the deployment with the generated config root selected before
   importing torchada.

## Step 1: Capture routing samples

`--topk-ids-dir` is required in every tuner invocation. Routing samples matter
because the expert-token distribution changes padding, the number of active
expert blocks, and the best `BLOCK_SIZE_M`.

The repository includes:

- [tuning_client.py](../src/torchada/triton/autotune/fused_moe/tuning_client.py),
  an OpenAI-compatible streaming request client; and
- [tuning_text.json](../src/torchada/triton/autotune/fused_moe/tuning_text.json),
  the long prompt used by that client.

The client drives the server but does not write route files itself. A temporary
server-side hook must save `topk_output.topk_ids` after the model computes its
routing decision.

### 1. Add a model save hook

For DeepSeek-style SGLang models, add the equivalent of the following directly
after routing in the MoE forward method. Use the tensor-parallel-rank helper
from the SGLang version being tested so that only rank 0 writes files.

```python
import os
import torch

# Import get_tensor_model_parallel_rank from the SGLang version being tested.

if hidden_states.shape[0] >= 4096 and get_tensor_model_parallel_rank() == 0:
    topk_ids_dir = os.environ["TORCHADA_TOPK_IDS_DIR"]
    os.makedirs(topk_ids_dir, exist_ok=True)
    if not hasattr(self, "save_idx"):
        self.save_idx = 0
    if self.save_idx <= 1:
        torch.save(
            topk_output.topk_ids.detach().cpu(),
            os.path.join(
                topk_ids_dir,
                f"topk_ids_layer{self.layer_id}_idx{self.save_idx}.pt",
            ),
        )
    self.save_idx += 1
```

For another model, replace `self.layer_id` and the location of `topk_output`
with that model's equivalents. The `hidden_states.shape[0] >= 4096` guard
avoids saving small decode steps. Adjust it when the target workload uses a
different prefill size.

Remove the hook after capture; it performs device-to-host copies and filesystem
writes and must not remain in a production server.

### 2. Launch the matching server without graph capture

Create the output directory and start SGLang with the same model, TP, EP,
quantization, and shared-expert settings that will be tuned:

```bash
export TORCHADA_TOPK_IDS_DIR=/tmp/torchada_topk_ids
mkdir -p "$TORCHADA_TOPK_IDS_DIR"

python -m sglang.launch_server \
    --model-path /models/DeepSeek-V2 \
    --tp-size 8 \
    --ep-size 8 \
    --port 8188 \
    --disable-cuda-graph
```

Keep `--disable-cuda-graph`: `torch.save` and other filesystem work cannot run
inside a captured CUDA/MUSA graph. On MUSA, use the project's normal
torchada-enabled launch wrapper if one is required by the deployment.

### 3. Send the bundled long request

```bash
python src/torchada/triton/autotune/fused_moe/tuning_client.py \
    --model auto \
    --ip 127.0.0.1 \
    --port 8188
```

`--model` defaults to `auto`; pass the model ID exposed by the server if its
OpenAI endpoint does not accept `auto`. `--ip` may point to a remote serving
host. Run the client more than once when samples from multiple requests are
needed.

### 4. Verify the captured files

The loader discovers every file matching
`topk_ids_layer*_idx*.pt`. Each file must contain a rank-2 tensor with shape
`[tokens, top_k]`:

```text
/tmp/torchada_topk_ids/
├── topk_ids_layer2_idx0.pt
├── topk_ids_layer2_idx1.pt
└── topk_ids_layer3_idx0.pt
```

Check the files before starting a long tuning run:

```bash
find "$TORCHADA_TOPK_IDS_DIR" -maxdepth 1 \
    -name 'topk_ids_layer*_idx*.pt' -print
```

The following read-only check prints each shape and expert-ID range:

```bash
TOPK_IDS_DIR="$TORCHADA_TOPK_IDS_DIR" python - <<'PY'
import glob
import os
import torch

pattern = os.path.join(os.environ["TOPK_IDS_DIR"], "topk_ids_layer*_idx*.pt")
for path in sorted(glob.glob(pattern)):
    ids = torch.load(path, map_location="cpu")
    print(path, tuple(ids.shape), f"min={ids.min().item()} max={ids.max().item()}")
PY
```

`--batch-size` means the MoE `M` dimension: the number of active tokens
entering the layer, not the number of user requests. Every captured sample
used by a tuning task must have:

- at least the requested number of rows; and
- at least the selected `top_k` number of columns.

If a sample is too small, tuning stops with a
`Captured top-k shape ... is smaller` error. Capture at least as many active
tokens as the largest batch size being tuned. Multiple layers or requests are
preferred: both eager and graph modes cycle through the loaded files in groups
of up to ten samples. Graph capture keeps separate routing metadata buffers
for those samples so replay measures the same route sequence as eager mode.

### EP expert-ID numbering

The torchada sep tuner passes captured IDs directly to the local kernel. The
IDs therefore must be in the range expected by the local expert tensor. This
differs from the referenced SGLang sep script, which converts its captured
global IDs to local IDs with `topk_ids // ep_size` during benchmarking.

If the SGLang hook above saves global, interleaved EP IDs, either make the hook
save the corresponding local IDs or preprocess the files before passing them
to torchada. Only apply the `// ep_size` conversion when the SGLang version and
expert mapping being tested use that convention; other expert-placement
strategies may need a different mapping.

Always verify that:

- all IDs are non-negative and smaller than the tuner's local expert count;
- shared-expert IDs are represented the same way in capture and tuning; and
- `--disable-shared-experts-fusion` and `--topk` match the captured route.

## Step 2: Tune configurations

Set an external config root to keep generated files separate from the source
tree:

```bash
export SGLANG_MOE_CONFIG_DIR="$PWD/generated_moe_configs"
```

### TP-only example

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe_sep.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 4 \
    --ep-size 1 \
    --batch-size 32,128,512 \
    --topk-ids-dir /tmp/torchada_topk_ids \
    --tune
```

### Combined TP and EP example

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe_sep.py \
    --model /models/DeepSeek-V2 \
    --tp-size 8 \
    --ep-size 4 \
    --dtype auto \
    --batch-size 1,8,32,128,512 \
    --topk-ids-dir /tmp/torchada_topk_ids_local \
    --tune
```

The route files in this EP example must already use the local expert numbering
described above.

### FP8 example

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe_sep.py \
    --model /models/DeepSeek-V3-FP8 \
    --tp-size 8 \
    --ep-size 8 \
    --dtype fp8_w8a8 \
    --batch-size 1,8,32,128 \
    --topk-ids-dir /tmp/torchada_topk_ids_local \
    --tune
```

### Multimodal MoE example

The script loads only the text/MoE configuration, so supported multimodal
families can be tuned in the same way:

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe_sep.py \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tp-size 2 \
    --ep-size 1 \
    --batch-size 32,128 \
    --topk-ids-dir /tmp/qwen3_vl_topk_ids \
    --tune
```

The accepted `--dtype` values are `auto`, `fp8_w8a8`, `int8_w8a8`, and
`int8_w8a16`. With `auto`, dtype and quantization mode are inferred from model
metadata.

> The sep implementation does not currently support `int4_w4a16`. Do not use
> `--dtype auto` with an INT4 model; use the standard tuner or add INT4 support
> to the sep path first.

### Batch-size selection and GPU parallelism

`--batch-size` may be repeated and each occurrence may contain comma-separated
values:

```bash
--batch-size 1,8,32 --batch-size 128 --batch-size 512
```

When omitted, the tuner uses:

```text
1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
1024, 1536, 2048, 3072, 4096
```

The multiprocessing scheduler assigns whole batch-size tasks to visible GPUs,
with one worker per GPU. It does not split the search space for a single batch
size across GPUs. Consequently, tuning several batch sizes can use multiple
GPUs concurrently, while a one-batch tuning command uses one GPU.

Restrict visible devices with the platform's normal visibility environment
variable when the host is shared or when tuning should use only one device.

### Limit the search space

The default search contains 396 candidates before model-specific filtering.
Use `--block-size-m` to retain only selected M tiles:

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe_sep.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --ep-size 1 \
    --batch-size 32,128 \
    --block-size-m 16,32,64 \
    --topk-ids-dir /tmp/torchada_topk_ids \
    --tune
```

For block-quantized models, the script additionally filters
`BLOCK_SIZE_K` candidates according to the model's quantization block width.

## Step 3: Benchmark one manual configuration

Without `--tune`, the script benchmarks a manually supplied configuration.
Provide exactly one batch size and six values in this order:

```text
BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M num_warps num_stages
```

Example:

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe_sep.py \
    --model /models/DeepSeek-V2 \
    --tp-size 8 \
    --ep-size 8 \
    --dtype auto \
    --batch-size 128 \
    --configs 64 64 128 16 8 1 \
    --topk-ids-dir /tmp/torchada_topk_ids_local
```

The output contains four timings in microseconds:

```text
t0       # kernel0 without TMA
t0_tma   # kernel0 with TMA
t1       # kernel1 without TMA
t1_tma   # kernel1 with TMA
```

This mode does not write a configuration file. Supplying multiple batch sizes
without `--tune` is unsupported.

## Generated files and runtime lookup

For Triton `3.2.0`, output is written below the selected config root:

```text
<SGLANG_MOE_CONFIG_DIR>/configs/triton_3_2_0/
├── E=...,N=...,device_name=...,dtype=....json
└── E=...,N=...,device_name=...,dtype=...._down.json
```

The normal file contains the selected `kernel0` configuration for each batch
size. The `_down.json` file contains the corresponding independently tuned
`kernel1` configuration. JSON keys are batch sizes represented as strings.

A down configuration may contain:

```json
{
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 16,
    "num_warps": 8,
    "num_stages": 1,
    "USE_TMA": true
}
```

`USE_TMA` is consumed by the down-projection runtime when TMA is available. If
the `_down.json` file is absent, runtime lookup falls back to the normal file,
which remains functional but may be slower.

`SGLANG_MOE_CONFIG_DIR` must point to the directory above `configs/`, not the
Triton-version directory. Export it before the serving process imports
torchada:

```bash
export SGLANG_MOE_CONFIG_DIR=/path/to/generated_moe_configs
python your_server_entrypoint.py
```

Unlike the upstream SGLang README workflow, no manual move into the SGLang
source tree is required when the runtime uses this environment variable.

If `SGLANG_MOE_CONFIG_DIR` is unset, the tuner writes into the installed or
source-tree `torchada/triton/autotune/fused_moe` directory. That location may
be read-only in a packaged environment, so an external config root is
recommended.

Set `TORCHADA_TUNE_MERGE_EXISTING_CONFIG=1` to preserve existing batch-size
entries while replacing entries tuned by the current command. Without it, the
output file is replaced by only the current run's batch sizes.

Multi-batch tuning also writes a timestamped
`tuning_result_YYYY-MM-DD HH:MM:SS.txt` in the current working directory. This
is a Python representation of the tuning results, not a benchmark JSON report.

## Graph, TMA, and profiling options

`--use-graph` passes graph mode directly to every benchmark worker and ranks
candidates with CUDA/MUSA graph replay. No graph-related environment variable
is required. Graph replay reduces Python launch overhead and is useful when the
serving workload also runs under graph capture.

The tuner treats TMA as disabled unless `TORCHADA_TUNE_DISABLE_TMA=0` is set.
To measure TMA on a build that provides `triton.set_allocator` and tensor
descriptors, use:

```bash
TORCHADA_TUNE_DISABLE_TMA=0 \
python src/torchada/triton/autotune/fused_moe/tune_moe_sep.py \
    --model /models/DeepSeek-V2 \
    --tp-size 8 \
    --ep-size 8 \
    --batch-size 128 \
    --topk-ids-dir /tmp/torchada_topk_ids_local \
    --use-graph \
    --tune
```

Leave TMA disabled when the installed MUSA Triton build exposes descriptor
types but lacks the allocator required to launch them. The setting applies to
both eager and graph measurement. When disabled, the reported TMA timing is
the matching non-TMA timing and generated down configs omit `USE_TMA`.

Set `NCU_ENABLE=1` only for kernel profiling. It reduces timing iterations to
one and is unsuitable for selecting stable production configurations.

## Parameters reference

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | Hugging Face/ModelScope model ID or local model path. |
| `--tp-size`, `--tp` | `2` | Total tensor-parallel world size. Must be divisible by EP size. |
| `--ep-size`, `--ep` | `1` | Expert-parallel world size used to derive local expert and intermediate shapes. |
| `--topk` | model config | Override the routed width when capture and model metadata differ. |
| `--dtype` | `auto` | `auto`, `fp8_w8a8`, `int8_w8a8`, or `int8_w8a16`. |
| `--seed` | `0` | Seed for random benchmark tensors and router weights. |
| `--batch-size` | default list | Active MoE token counts; repeatable and comma-separated. |
| `--tune` | off | Search candidates and write normal/down config files. |
| `--disable-shared-experts-fusion` | off | Exclude fused shared experts; must match capture and deployment. |
| `--configs` | none | Six integers for manual single-config benchmark mode. |
| `--topk-ids-dir` | required | Directory containing `topk_ids_layer*_idx*.pt`. |
| `--use-graph` | off | Rank candidates using CUDA/MUSA graph replay. |
| `--block-size-m` | all | Comma-separated `BLOCK_SIZE_M` filter, for example `16,32,64`. |

## Environment variables

| Variable | Effect |
| --- | --- |
| `TORCHADA_TOPK_IDS_DIR` | Example capture-hook destination; the tuner itself receives the directory through `--topk-ids-dir`. |
| `SGLANG_MOE_CONFIG_DIR` | Config root above `configs/triton_<version>/` for both writing and runtime lookup. |
| `TORCHADA_TUNE_MERGE_EXISTING_CONFIG=1` | Merge new batch-size entries into existing normal/down files. |
| `TORCHADA_TUNE_DISABLE_TMA` | `1` disables TMA in eager and graph modes; the current default is disabled. Set `0` only when TMA descriptors and allocator are usable. |
| `NCU_ENABLE=1` | Reduce timing iterations for profiler collection. |

## Troubleshooting

### `No captured top-k files found`

Check that `--topk-ids-dir` directly contains files matching
`topk_ids_layer*_idx*.pt`, the model hook was reached, and the graph-disabled
request completed before starting the tuner.

### `Captured top-k shape ... is smaller`

Capture a request with at least the largest requested active-token count and
enough top-k columns. Alternatively, tune a smaller batch-size list or use
`--topk` to select the captured route width.

### Kernel failure or invalid results with EP

Inspect the captured minimum and maximum IDs. The most common cause is passing
global expert IDs to a tuner that allocated only local experts. Convert them
according to the exact expert-placement convention used by the capture server,
then confirm all IDs fit the local range.

### Every candidate fails

Verify model shape, dtype, quantization block shape, TP/EP divisibility,
captured ID range, and the availability of `sgl_kernel` or `vllm` alignment
ops. A candidate that raises Triton out-of-resource, runtime, or assertion
errors is skipped; if every candidate is skipped, no config can be selected.

### TMA allocator errors

Disable TMA for either eager or graph mode:

```bash
TORCHADA_TUNE_DISABLE_TMA=1 \
python src/torchada/triton/autotune/fused_moe/tune_moe_sep.py \
    ... \
    --tune
```

### Generated files are not used

Check all filename selectors: Triton version, device name, dtype, block shape,
`E`, and `N`. Confirm that the serving process receives the same
`SGLANG_MOE_CONFIG_DIR`, TP/EP values, dtype, and shared-expert settings before
it imports torchada.

### Only one GPU is busy

Parallelism is at batch-size-task granularity. A command that tunes only one
batch size uses one worker for the actual search. Provide several batch sizes
to distribute work across multiple visible GPUs.

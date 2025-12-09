"""
Profile Expert Parallel with All2All communication (dispatch/combine).

Usage:
    # 2 GPU, DP=2, TP=1 (pure EP, requires 2 GPUs)
    CUDA_VISIBLE_DEVICES=2,3 uv run python profile_ep_all2all.py --dp-size=2 --tp-size=1

    # 4 GPU, DP=2, TP=2 (EP + TP, requires 4 GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python profile_ep_all2all.py --dp-size=2 --tp-size=2

This script triggers real EP all2all communication:
    - dispatch: all-gather (collect tokens from all ranks)
    - compute: fused_moe_kernel (MoE computation)
    - combine: reduce-scatter (distribute results back)
"""

import os
from time import sleep
from multiprocessing import Process

# Set profiler output directory
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./ep_profile_all2all"

# Set all2all backend
os.environ["VLLM_ALL2ALL_BACKEND"] = "allgather_reducescatter"

from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Profile EP All2All")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Thinking-2507",
        help="Model name or path",
    )
    parser.add_argument("--dp-size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph for cleaner profile",
    )
    return parser.parse_args()


def main(
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    tp_size,
    gpu_memory_utilization,
    max_model_len,
    enforce_eager,
):
    # Set DP environment variables
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Test prompts
    prompts = ["Hello, my name is"] * 20

    # Each DP rank processes different prompts
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    prompts = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
    if len(prompts) == 0:
        prompts = ["Placeholder"]

    print(f"DP rank {global_dp_rank} processing {len(prompts)} prompts")

    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

    # Create LLM
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=True,  # Enable EP
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )

    # Start profiling
    print(f"DP rank {global_dp_rank}: Starting profile...")
    llm.start_profile()

    # Run inference
    outputs = llm.generate(prompts, sampling_params)

    # Stop profiling
    llm.stop_profile()
    print(f"DP rank {global_dp_rank}: Profile stopped")

    # Print partial outputs
    for i, output in enumerate(outputs[:2]):
        print(
            f"DP rank {global_dp_rank}, Prompt: {output.prompt!r}, "
            f"Generated: {output.outputs[0].text[:50]!r}..."
        )

    sleep(2)


if __name__ == "__main__":
    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size

    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    print("=" * 60)
    print("Profile Expert Parallel with All2All")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"DP size: {dp_size}, TP size: {tp_size}")
    print(f"Total GPUs needed: {dp_size * tp_size}")
    print(f"Profile output: ./ep_profile_all2all/")
    print(f"All2All backend: {os.environ.get('VLLM_ALL2ALL_BACKEND', 'default')}")
    print("=" * 60)

    procs = []
    for local_dp_rank in range(dp_size):
        global_dp_rank = local_dp_rank
        proc = Process(
            target=main,
            args=(
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                args.gpu_memory_utilization,
                args.max_model_len,
                args.enforce_eager,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=300)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid}")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("Profile completed!")
        print("=" * 60)
        print("Profile files saved to: ./ep_profile_all2all/")
        print("\nAnalyzing profile results:")
        print("  - dispatch: look for all_gatherv / naive_multicast")
        print("  - compute: look for fused_moe_kernel")
        print("  - combine: look for reduce_scatterv / all_reduce")
        print("\nExpected dispatch:compute:combine ratio is approximately 1:1:1")

    exit(exit_code)

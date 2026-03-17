"""Triton kernels for Shuffled Reduce-Scatter (SRS) and Shuffled AllGather (SAG).

These implement the fused communication primitives from the Sem-MoE paper,
using symmetric memory P2P writes for cross-GPU data movement.

SRS replaces post-attention all_reduce with: shuffle + reduce_scatter (fused).
SAG replaces post-MoE allgather with: allgather + unshuffle (fused).
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


class SemMoeSRSPool:
    """Manages symmetric memory buffers for SRS/SAG communication."""

    def __init__(self) -> None:
        self.is_initialized = False
        self.buf: torch.Tensor | None = None
        self.hdl = None
        self.bufs: tuple[torch.Tensor, ...] | None = None
        # Views into the symmetric memory for SRS and SAG
        self.peer_srs_bufs: tuple[torch.Tensor, ...] | None = None
        self.peer_sag_bufs: tuple[torch.Tensor, ...] | None = None
        self._srs_region_size = 0
        self._sag_region_size = 0

    def initialize(
        self,
        max_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        num_ranks: int,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        if self.is_initialized:
            return

        elem_size = torch.empty((), dtype=dtype).element_size()

        # SRS recv buffer: each rank stores partial data from all source ranks
        # Layout: [num_ranks * max_tokens * hidden_dim]
        self._srs_region_size = num_ranks * max_tokens * hidden_dim * elem_size
        # SAG output buffer: each rank's full output [max_tokens * hidden_dim]
        self._sag_region_size = max_tokens * hidden_dim * elem_size

        total_size = self._srs_region_size + self._sag_region_size
        # Align to 128 bytes
        total_size = ((total_size + 127) // 128) * 128

        self.buf = symm_mem.empty(
            (total_size,), dtype=torch.uint8, device=device
        )
        self.hdl = symm_mem.rendezvous(self.buf, group=group)
        self.bufs = tuple(
            self.hdl.get_buffer(r, self.buf.shape, self.buf.dtype)
            for r in range(num_ranks)
        )
        self.hdl.barrier(channel=0)

        self._dtype = dtype
        self._hidden_dim = hidden_dim
        self._max_tokens = max_tokens
        self._num_ranks = num_ranks
        self._elem_size = elem_size
        self.is_initialized = True

        logger.info(
            "SemMoeSRSPool initialized: max_tokens=%d, hidden_dim=%d, "
            "num_ranks=%d, total_symm_mem=%.1fMB",
            max_tokens,
            hidden_dim,
            num_ranks,
            total_size / 1024 / 1024,
        )

    def get_srs_recv_bufs(
        self,
        num_ranks: int,
        chunk_size: int,
        hidden_dim: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        """Get per-rank SRS receive buffers as shaped tensors."""
        assert self.is_initialized
        result = []
        for buf in self.bufs:
            storage = buf.untyped_storage()
            t = torch.empty(0, dtype=dtype, device=buf.device)
            t.set_(
                storage,
                0,
                torch.Size([num_ranks, chunk_size, hidden_dim]),
            )
            result.append(t)
        return tuple(result)

    def get_sag_out_bufs(
        self,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        """Get per-rank SAG output buffers as shaped tensors."""
        assert self.is_initialized
        srs_offset = self._srs_region_size
        elem_size = torch.empty((), dtype=dtype).element_size()
        result = []
        for buf in self.bufs:
            storage = buf.untyped_storage()
            t = torch.empty(0, dtype=dtype, device=buf.device)
            t.set_(
                storage,
                srs_offset // elem_size,
                torch.Size([num_tokens, hidden_dim]),
            )
            result.append(t)
        return tuple(result)


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _srs_scatter_kernel(
    peer_dst_ptrs,  # peer SRS recv buffer pointers (tuple)
    src_ptr,  # [N, D] local partial hidden_states (un-shuffled)
    shf_idx_ptr,  # [N] shuffle indices
    src_stride_m,  # stride of src along token dim (= D)
    dst_stride_slot,  # stride between source-rank slots (= chunk_size * D)
    dst_stride_m,  # stride along token dim in dst (= D)
    chunk_size,  # N / num_ranks (fixed)
    D: tl.constexpr,  # hidden_dim
    SRC_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused shuffle + P2P scatter write.

    Each program handles one shuffled position (pid in 0..N-1).
    Reads src[shf_idx[pid], :] and writes to the appropriate rank's buffer
    at the correct offset.
    """
    pid = tl.program_id(0)  # shuffled position index (0..N-1)
    dst_rank = pid // chunk_size
    dst_offset = pid % chunk_size

    # Read shuffle index: which original token goes to this shuffled position
    src_token = tl.load(shf_idx_ptr + pid)

    # Select target rank's buffer pointer
    dst_ptr = tl.zeros((1,), dtype=tl.int64).item()
    for r in tl.static_range(N_RANKS):
        if dst_rank == r:
            dst_ptr = peer_dst_ptrs[r].to(tl.int64, bitcast=True)
    dst_ptr = tl.multiple_of(dst_ptr.to(src_ptr.dtype, bitcast=True), 16)

    # Fused shuffle + scatter: read src[src_token, :], write dst[SRC_RANK, dst_offset, :]
    offs_d = tl.arange(0, BLOCK_D)
    for start_d in range(0, D, BLOCK_D):
        mask = start_d + offs_d < D
        src_data = tl.load(
            src_ptr + src_token * src_stride_m + start_d + offs_d, mask=mask
        )
        dst_addr = (
            dst_ptr
            + SRC_RANK * dst_stride_slot
            + dst_offset * dst_stride_m
            + start_d
            + offs_d
        )
        tl.store(dst_addr, src_data, mask=mask)


@triton.jit
def _srs_reduce_kernel(
    output_ptr,  # [chunk_size, D] output
    recv_buf_ptr,  # [N_RANKS, chunk_size, D] received partials
    chunk_size,  # fixed chunk size
    stride_slot,  # = chunk_size * D (source-rank stride)
    stride_m,  # = D (token stride)
    D: tl.constexpr,
    N_RANKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Local reduce: sum partials from all source ranks."""
    pid = tl.program_id(0)  # chunk-internal token index
    offs_d = tl.arange(0, BLOCK_D)
    for start_d in range(0, D, BLOCK_D):
        mask = start_d + offs_d < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for r in tl.static_range(N_RANKS):
            src = tl.load(
                recv_buf_ptr + r * stride_slot + pid * stride_m + start_d + offs_d,
                mask=mask,
            )
            acc += src.to(tl.float32)
        tl.store(
            output_ptr + pid * stride_m + start_d + offs_d,
            acc.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _sag_kernel(
    peer_dst_ptrs,  # peer SAG output buffer pointers (tuple)
    src_ptr,  # [chunk_size, D] local MoE output
    inv_shf_ptr,  # [N] inverse shuffle (shuffled_pos -> original_pos)
    chunk_offset,  # rank * chunk_size
    src_stride_m,  # stride along token dim in src
    dst_stride_m,  # stride along token dim in dst
    chunk_size,  # local chunk size
    D: tl.constexpr,
    SRC_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused allgather + unshuffle via P2P writes.

    Each rank writes its chunk to all peers' output buffers at the
    original (un-shuffled) positions. This eliminates a separate unshuffle step.
    """
    pid = tl.program_id(0)  # chunk-internal token index
    if pid >= chunk_size:
        return

    # shuffled_pos -> original_pos
    shuffled_pos = chunk_offset + pid
    original_pos = tl.load(inv_shf_ptr + shuffled_pos)

    offs_d = tl.arange(0, BLOCK_D)
    for start_d in range(0, D, BLOCK_D):
        mask = start_d + offs_d < D
        src_data = tl.load(
            src_ptr + pid * src_stride_m + start_d + offs_d, mask=mask
        )
        # Write to all ranks' output buffers at original_pos
        for r in tl.static_range(N_RANKS):
            dst_ptr = peer_dst_ptrs[r].to(tl.int64, bitcast=True)
            dst_ptr = tl.multiple_of(dst_ptr.to(src_ptr.dtype, bitcast=True), 16)
            dst_addr = dst_ptr + original_pos * dst_stride_m + start_d + offs_d
            tl.store(dst_addr, src_data, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def srs_forward(
    pool: SemMoeSRSPool,
    hidden_states: torch.Tensor,  # [N, D] partial (un-shuffled)
    shf_idx: torch.Tensor,  # [N] shuffle indices
    chunk_size: int,  # = N / num_ranks (fixed)
    rank: int,
    num_ranks: int,
) -> torch.Tensor:
    """SRS: fused shuffle + P2P scatter + local reduce -> [chunk_size, D]."""
    N, D = hidden_states.shape

    # Get per-rank receive buffers and zero them
    srs_bufs = pool.get_srs_recv_bufs(num_ranks, chunk_size, D, hidden_states.dtype)
    srs_bufs[rank].zero_()

    BLOCK_D = min(512, triton.next_power_of_2(D))

    # Phase 1: Fused shuffle + scatter
    _srs_scatter_kernel[(N,)](
        tuple(srs_bufs),
        hidden_states,
        shf_idx,
        hidden_states.stride(0),
        chunk_size * D,  # dst_stride_slot
        D,  # dst_stride_m
        chunk_size,
        D=D,
        SRC_RANK=rank,
        N_RANKS=num_ranks,
        BLOCK_D=BLOCK_D,
    )
    pool.hdl.barrier(channel=0)

    # Phase 2: Local reduce
    output = torch.empty(
        chunk_size, D, dtype=hidden_states.dtype, device=hidden_states.device
    )
    _srs_reduce_kernel[(chunk_size,)](
        output,
        srs_bufs[rank],
        chunk_size,
        chunk_size * D,  # stride_slot
        D,  # stride_m
        D=D,
        N_RANKS=num_ranks,
        BLOCK_D=BLOCK_D,
    )
    return output


def sag_forward(
    pool: SemMoeSRSPool,
    chunk: torch.Tensor,  # [chunk_size, D]
    inv_shf: torch.Tensor,  # [N]
    chunk_size: int,  # = N / num_ranks (fixed)
    rank: int,
    num_ranks: int,
) -> torch.Tensor:
    """SAG: broadcast chunk to all ranks + unshuffle -> [N, D]."""
    N = chunk_size * num_ranks
    D = chunk.shape[1]
    chunk_offset = rank * chunk_size

    # Get per-rank output buffers
    sag_bufs = pool.get_sag_out_bufs(N, D, chunk.dtype)

    BLOCK_D = min(512, triton.next_power_of_2(D))

    _sag_kernel[(chunk_size,)](
        tuple(sag_bufs),
        chunk,
        inv_shf,
        chunk_offset,
        chunk.stride(0),
        D,  # dst_stride_m
        chunk_size,
        D=D,
        SRC_RANK=rank,
        N_RANKS=num_ranks,
        BLOCK_D=BLOCK_D,
    )
    pool.hdl.barrier(channel=0)

    # Return this rank's output (already in original order due to inv_shf)
    return sag_bufs[rank][:N]


# ---------------------------------------------------------------------------
# Fallback wrappers (NCCL-based)
# ---------------------------------------------------------------------------


def srs_nccl_fallback(
    hidden_states: torch.Tensor,  # [N, D] partial (un-shuffled)
    shf_idx: torch.Tensor,  # [N] shuffle indices
    chunk_size: int,
    tp_group: object,  # GroupCoordinator
) -> torch.Tensor:
    """NCCL fallback: shuffle + standard reduce_scatter."""
    shuffled = hidden_states[shf_idx]
    return tp_group.reduce_scatter(shuffled, dim=0)


def sag_nccl_fallback(
    chunk: torch.Tensor,  # [chunk_size, D]
    inv_shf: torch.Tensor,  # [N]
    original_num_tokens: int,
    tp_group: object,  # GroupCoordinator
) -> torch.Tensor:
    """NCCL fallback: standard allgather + unshuffle."""
    gathered = tp_group.all_gather(chunk, dim=0)
    # Unshuffle and truncate padding
    return gathered[inv_shf][:original_num_tokens]


# ---------------------------------------------------------------------------
# Unified entry points
# ---------------------------------------------------------------------------

import os

_SRS_BACKEND = os.environ.get("SEM_MOE_SRS_BACKEND", "triton").lower()


def srs_or_fallback(
    hidden_states: torch.Tensor,
    shf_idx: torch.Tensor,
    chunk_size: int,
    pool: SemMoeSRSPool | None,
    rank: int,
    num_ranks: int,
    tp_group: object | None = None,
) -> torch.Tensor:
    """SRS with Triton, fallback to NCCL if symmetric memory unavailable."""
    if _SRS_BACKEND != "nccl" and pool is not None and pool.is_initialized:
        return srs_forward(pool, hidden_states, shf_idx, chunk_size, rank, num_ranks)
    elif tp_group is not None:
        return srs_nccl_fallback(hidden_states, shf_idx, chunk_size, tp_group)
    else:
        raise RuntimeError("Neither SRS pool nor TP group available for reduce_scatter")


def sag_or_fallback(
    chunk: torch.Tensor,
    inv_shf: torch.Tensor,
    chunk_size: int,
    original_num_tokens: int,
    pool: SemMoeSRSPool | None,
    rank: int,
    num_ranks: int,
    tp_group: object | None = None,
) -> torch.Tensor:
    """SAG with Triton, fallback to NCCL if symmetric memory unavailable."""
    if _SRS_BACKEND != "nccl" and pool is not None and pool.is_initialized:
        return sag_forward(pool, chunk, inv_shf, chunk_size, rank, num_ranks)
    elif tp_group is not None:
        return sag_nccl_fallback(chunk, inv_shf, original_num_tokens, tp_group)
    else:
        raise RuntimeError("Neither SRS pool nor TP group available for allgather")

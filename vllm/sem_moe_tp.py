from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

# Minimum number of tokens to apply rebatching.
# Below this threshold, the overhead exceeds the benefit.
import os as _os
_MIN_REBATCH_TOKENS = int(_os.getenv("SEM_MOE_MIN_REBATCH_TOKENS", "16"))


@dataclass
class SemMoeTPLayerTensors:
    """Per-layer GPU tensors for TP token scheduling."""

    t_full: torch.Tensor  # [vocab_size] int32 — predicted target device per token
    tp_full: torch.Tensor  # [vocab_size] float32 — confidence of T prediction
    a_table: torch.Tensor  # [num_devices^lookback] int32 — device-seq prediction
    ap_table: torch.Tensor  # [num_devices^lookback] float32 — confidence of A prediction


@dataclass
class SemMoeTPContext:
    """Mutable context for TP token rebatching across MoE layers."""

    layer_tensors: dict[int, SemMoeTPLayerTensors]
    lookback: int
    num_devices: int
    moe_layer_ids: tuple[int, ...]

    # Maps layer_id -> index in moe_layer_ids for device_trace column indexing
    _layer_id_to_moe_idx: dict[int, int] = field(default_factory=dict, repr=False)

    # Mutable per-forward state
    device_trace: torch.Tensor | None = field(default=None, repr=False)
    moe_layer_counter: int = 0

    def __post_init__(self) -> None:
        self._layer_id_to_moe_idx = {
            lid: idx for idx, lid in enumerate(self.moe_layer_ids)
        }

    def reset(self, num_tokens: int, device: torch.device) -> None:
        """Reset per-forward mutable state. Called at the start of each forward pass."""
        num_moe_layers = len(self.moe_layer_ids)
        if (
            self.device_trace is None
            or self.device_trace.shape[0] < num_tokens
            or self.device_trace.shape[1] != num_moe_layers
            or self.device_trace.device != device
        ):
            self.device_trace = torch.zeros(
                num_tokens, num_moe_layers, dtype=torch.int32, device=device
            )
        else:
            self.device_trace[:num_tokens].zero_()
        self.moe_layer_counter = 0

    def moe_index(self, layer_id: int) -> int:
        return self._layer_id_to_moe_idx[layer_id]


def build_tp_context(
    schedule: object,  # SemMoeSchedule (avoid circular import)
    device: torch.device,
) -> SemMoeTPContext | None:
    """Move per-layer T/Tp/A/Ap tables to GPU. Called once at model finalization."""
    layer_tensors: dict[int, SemMoeTPLayerTensors] = {}

    for layer_id in schedule.layer_ids:  # type: ignore[attr-defined]
        ls = schedule.layers[layer_id]  # type: ignore[attr-defined]
        if ls.t_full is None or ls.tp_full is None:
            logger.warning(
                "Sem-MoE TP tables missing for layer %s; skipping TP context.", layer_id
            )
            return None
        lt = SemMoeTPLayerTensors(
            t_full=torch.from_numpy(ls.t_full).to(device),
            tp_full=torch.from_numpy(ls.tp_full).to(device),
            a_table=(
                torch.from_numpy(ls.a_table).to(device)
                if ls.a_table is not None
                else torch.zeros(1, dtype=torch.int32, device=device)
            ),
            ap_table=(
                torch.from_numpy(ls.ap_table).to(device)
                if ls.ap_table is not None
                else torch.zeros(1, dtype=torch.float32, device=device)
            ),
        )
        layer_tensors[layer_id] = lt

    if not layer_tensors:
        return None

    ctx = SemMoeTPContext(
        layer_tensors=layer_tensors,
        lookback=schedule.lookback,  # type: ignore[attr-defined]
        num_devices=schedule.num_devices,  # type: ignore[attr-defined]
        moe_layer_ids=schedule.layer_ids,  # type: ignore[attr-defined]
    )
    logger.info(
        "Built Sem-MoE TP context: %d layers, lookback=%d, num_devices=%d",
        len(layer_tensors),
        ctx.lookback,
        ctx.num_devices,
    )
    return ctx


def encode_device_sequence_batch(
    trace_slice: torch.Tensor,
    num_devices: int,
) -> torch.Tensor:
    """Encode device trace columns into a single sequence index (vectorized).

    For lookback=2, num_devices=2:
        seq_id = trace[:, 0] * 2 + trace[:, 1]

    Args:
        trace_slice: [num_tokens, lookback] int32
        num_devices: number of TP devices

    Returns:
        [num_tokens] int64 — encoded sequence IDs for A-table lookup
    """
    result = torch.zeros(
        trace_slice.shape[0], dtype=torch.long, device=trace_slice.device
    )
    for i in range(trace_slice.shape[1]):
        result = result * num_devices + trace_slice[:, i].long()
    return result


def rebatch_for_layer(
    ctx: SemMoeTPContext,
    layer_id: int,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    indices_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute shuffle indices for TP token rebatching (Algorithm 3).

    Steps:
        1. T-table lookup: predict target device per token via token identity
        2. A-table lookup: predict target device via inter-layer device sequence
        3. Confidence comparison: pick whichever predictor is more confident
        4. argsort + align: produce equal-sized chunks for reduce_scatter

    Args:
        ctx: TP context with per-layer tables and device trace
        layer_id: current MoE layer id
        input_ids: [num_tokens] int64
        hidden_states: [num_tokens, hidden_dim]
        indices_only: if True, skip hidden_states shuffle (SRS path computes
            it separately). First return value will be the un-shuffled
            hidden_states (possibly padded).

    Returns:
        (hidden_states_shuffled, shf_idx, inv_shf, chunk_size)
        - hidden_states_shuffled: [N_padded, hidden_dim] shuffled (or padded-only if indices_only)
        - shf_idx: [N_padded] shuffle permutation
        - inv_shf: [N_padded] inverse permutation for unshuffle
        - chunk_size: N_padded // num_devices (fixed per chunk)
    """
    num_tokens = hidden_states.shape[0]
    device = hidden_states.device
    lt = ctx.layer_tensors[layer_id]
    moe_idx = ctx.moe_index(layer_id)

    # --- Step 1: T-table (token-identity prediction) ---
    # Clamp input_ids to valid range for the lookup table
    clamped_ids = input_ids.clamp(0, lt.t_full.shape[0] - 1)
    dev_ids_T = lt.t_full[clamped_ids]  # [num_tokens] int32
    conf_T = lt.tp_full[clamped_ids]  # [num_tokens] float32

    # --- Step 2: A-table (device-sequence prediction) ---
    if ctx.moe_layer_counter >= ctx.lookback and lt.a_table.shape[0] > 1:
        start = ctx.moe_layer_counter - ctx.lookback
        assert ctx.device_trace is not None
        trace_slice = ctx.device_trace[
            :num_tokens, start : ctx.moe_layer_counter
        ]  # [num_tokens, lookback]
        seq_ids = encode_device_sequence_batch(trace_slice, ctx.num_devices)
        # Clamp to valid A-table range
        seq_ids = seq_ids.clamp(0, lt.a_table.shape[0] - 1)
        dev_ids_A = lt.a_table[seq_ids]  # [num_tokens]
        conf_A = lt.ap_table[seq_ids]  # [num_tokens]
        # Pick whichever predictor is more confident
        dev_ids = torch.where(conf_T > conf_A, dev_ids_T, dev_ids_A)
    else:
        dev_ids = dev_ids_T

    # --- Step 3: Record device trace (predicted dev_ids, not aligned) ---
    assert ctx.device_trace is not None
    ctx.device_trace[:num_tokens, moe_idx] = dev_ids.int()
    ctx.moe_layer_counter += 1

    # --- Step 4: argsort + align (equal chunk sizes) ---
    # Pad to nearest multiple of num_devices if needed
    n_padded = num_tokens
    if num_tokens % ctx.num_devices != 0:
        pad_n = ctx.num_devices - (num_tokens % ctx.num_devices)
        n_padded = num_tokens + pad_n
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_n))
        # Padding tokens get device id = num_devices - 1 (last device)
        # so they sort to the end
        dev_ids = torch.cat(
            [
                dev_ids,
                torch.full(
                    (pad_n,), ctx.num_devices - 1, dtype=dev_ids.dtype, device=device
                ),
            ]
        )

    # Stable argsort groups tokens by predicted device, preserving relative order
    shf_idx = torch.argsort(dev_ids.long(), stable=True)
    # O(N) inverse permutation via scatter (instead of O(N log N) argsort)
    inv_shf = torch.empty_like(shf_idx)
    inv_shf[shf_idx] = torch.arange(n_padded, device=device, dtype=shf_idx.dtype)
    chunk_size = n_padded // ctx.num_devices

    # Shuffle hidden states (skip when caller only needs indices, e.g. SRS path)
    if not indices_only:
        hidden_states = hidden_states[shf_idx]

    return hidden_states, shf_idx, inv_shf, chunk_size


def unshuffle_output(
    output: torch.Tensor,
    inv_shf: torch.Tensor,
    original_num_tokens: int,
) -> torch.Tensor:
    """Restore original token order after MoE processing.

    Args:
        output: [N_padded, hidden_dim] in shuffled order
        inv_shf: [N_padded] inverse permutation
        original_num_tokens: number of real (non-padding) tokens

    Returns:
        [original_num_tokens, hidden_dim] in original order
    """
    return output[inv_shf][:original_num_tokens]

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

SEM_MOE_ENV = "SEM_MOE"
SEM_MOE_TABLES_ENV = "SEM_MOE_TABLES"
SEM_MOE_MODE_ENV = "SEM_MOE_MODE"
SEM_MOE_DEBUG_FALLBACK_ENV = "SEM_MOE_DEBUG_FALLBACK"

SUPPORTED_MODES = {"dp", "tp", "both"}
SUPPORTED_BLOCK_TYPES = {"Qwen3NextSparseMoeBlock", "Qwen3MoeSparseMoeBlock"}


@dataclass(frozen=True)
class SemMoeConfig:
    enabled: bool
    schedule_dir: Path | None
    mode: str
    debug_fallback: bool


@dataclass(frozen=True)
class SemMoeLayerSchedule:
    layer_id: int
    num_experts: int
    num_devices: int
    expert_labels: np.ndarray
    expert_permutation: np.ndarray
    expert_inverse_permutation: np.ndarray
    gate_permutation: np.ndarray
    gate_inverse_permutation: np.ndarray
    score_full: np.ndarray
    # TP tables (only loaded when mode in {"tp", "both"})
    t_full: np.ndarray | None = None        # [vocab_size] int32
    tp_full: np.ndarray | None = None       # [vocab_size] float32
    a_table: np.ndarray | None = None       # [num_devices^lookback] int32
    ap_table: np.ndarray | None = None      # [num_devices^lookback] float32


@dataclass(frozen=True)
class SemMoeSchedule:
    schedule_dir: Path
    num_devices: int
    layer_ids: tuple[int, ...]
    layers: dict[int, SemMoeLayerSchedule]
    dp_score_full: np.ndarray
    lookback: int = 2


@dataclass
class SemMoeBoundLayer:
    layer_id: int
    experts: torch.nn.Module
    gate: torch.nn.Module
    schedule: SemMoeLayerSchedule
    loader_map: torch.Tensor
    runtime_map: torch.Tensor


@dataclass
class SemMoeModelContext:
    schedule: SemMoeSchedule
    bound_layers: list[SemMoeBoundLayer]
    applied: bool = False


def sem_moe_config() -> SemMoeConfig:
    enabled = _env_flag(SEM_MOE_ENV)
    raw_schedule_dir = os.getenv(SEM_MOE_TABLES_ENV)
    schedule_dir = _resolve_schedule_dir(raw_schedule_dir)
    if enabled and schedule_dir is None:
        logger.warning(
            "Sem-MoE is enabled but %s=%r does not point to a valid schedule directory.",
            SEM_MOE_TABLES_ENV,
            raw_schedule_dir,
        )
    mode = (os.getenv(SEM_MOE_MODE_ENV) or "both").strip().lower()
    if mode not in SUPPORTED_MODES:
        logger.warning(
            "Ignoring unsupported %s=%r; expected one of %s.",
            SEM_MOE_MODE_ENV,
            mode,
            sorted(SUPPORTED_MODES),
        )
        mode = "both"
    return SemMoeConfig(
        enabled=enabled and schedule_dir is not None,
        schedule_dir=schedule_dir,
        mode=mode,
        debug_fallback=_env_flag(SEM_MOE_DEBUG_FALLBACK_ENV),
    )


def sem_moe_dp_enabled() -> bool:
    config = sem_moe_config()
    return config.enabled and config.mode in {"dp", "both"}


def sem_moe_tp_enabled() -> bool:
    config = sem_moe_config()
    return config.enabled and config.mode in {"tp", "both"}


def sem_moe_srs_enabled() -> bool:
    """True when SRS/SAG kernels should replace standard all_reduce.

    This is the full Milestone 3 path: o_proj skips reduce, decoder layer
    handles SRS after attention + SAG after MoE.  When debug_fallback is on,
    we keep standard all_reduce and only shuffle/unshuffle inside the MoE block.
    """
    config = sem_moe_config()
    return config.enabled and config.mode in {"tp", "both"} and not config.debug_fallback


def clear_sem_moe_caches() -> None:
    _load_schedule_cached.cache_clear()


def load_sem_moe_schedule() -> SemMoeSchedule | None:
    config = sem_moe_config()
    if not config.enabled or config.schedule_dir is None:
        return None
    return _load_schedule_cached(str(config.schedule_dir), config.debug_fallback)


@lru_cache(maxsize=8)
def _load_schedule_cached(
    schedule_dir_text: str,
    debug_fallback: bool,
) -> SemMoeSchedule | None:
    schedule_dir = Path(schedule_dir_text)
    manifest_path = schedule_dir / "manifest.json"
    if not manifest_path.exists():
        logger.warning(
            "Sem-MoE enabled but manifest is missing under %s.",
            schedule_dir,
        )
        return None

    manifest = _load_json(manifest_path)
    try:
        num_devices = int(manifest["num_devices"])
        layer_ids = tuple(int(layer_id) for layer_id in manifest["moe_layer_ids"])
    except KeyError as exc:
        logger.warning("Invalid Sem-MoE manifest %s: missing %s.", manifest_path, exc)
        return None

    lookback = int(manifest.get("lookback", 2))
    mode = (os.getenv(SEM_MOE_MODE_ENV) or "both").strip().lower()
    load_tp_tables = mode in {"tp", "both"}

    layers: dict[int, SemMoeLayerSchedule] = {}
    dp_score_full: np.ndarray | None = None
    for layer_id in layer_ids:
        layer_path = schedule_dir / f"semmoe_layer{layer_id}.npz"
        if not layer_path.exists():
            logger.warning("Sem-MoE layer artifact is missing: %s.", layer_path)
            return None
        with np.load(layer_path) as payload:
            try:
                expert_labels = payload["E"].astype(np.int32, copy=True)
                expert_permutation = payload["expert_permutation"].astype(
                    np.int32, copy=True
                )
                expert_inverse_permutation = payload["expert_inverse_permutation"].astype(
                    np.int32, copy=True
                )
                gate_permutation = payload.get(
                    "gating_column_permutation", expert_permutation
                )
                gate_inverse_permutation = payload.get(
                    "gating_column_inverse_permutation",
                    expert_inverse_permutation,
                )
                # DEBUG: force identity permutation to isolate issue
                if _env_flag("SEM_MOE_IDENTITY_PERM"):
                    n = len(expert_permutation)
                    expert_permutation = np.arange(n, dtype=np.int32)
                    expert_inverse_permutation = np.arange(n, dtype=np.int32)
                    gate_permutation = np.arange(n, dtype=np.int32)
                    gate_inverse_permutation = np.arange(n, dtype=np.int32)
                score_full = _load_layer_score_full(payload, num_devices, debug_fallback)
                t_full = None
                tp_full = None
                a_table = None
                ap_table = None
                if load_tp_tables:
                    if "T_full" in payload:
                        t_full = payload["T_full"].astype(np.int32, copy=True)
                    if "Tp_full" in payload:
                        tp_full = payload["Tp_full"].astype(np.float32, copy=True)
                    if "A" in payload:
                        a_table = payload["A"].astype(np.int32, copy=True)
                    if "Ap" in payload:
                        ap_table = payload["Ap"].astype(np.float32, copy=True)
            except (KeyError, ValueError) as exc:
                logger.warning("Invalid Sem-MoE layer artifact %s: %s", layer_path, exc)
                return None

        if expert_labels.shape != (expert_permutation.shape[0],):
            logger.warning(
                "Invalid Sem-MoE layer artifact %s: mismatched expert shapes.",
                layer_path,
            )
            return None
        if score_full.shape[1] != num_devices:
            logger.warning(
                "Invalid Sem-MoE layer artifact %s: score table width %s != num_devices %s.",
                layer_path,
                score_full.shape[1],
                num_devices,
            )
            return None
        if dp_score_full is None:
            dp_score_full = np.zeros_like(score_full, dtype=np.float32)
        elif dp_score_full.shape != score_full.shape:
            logger.warning(
                "Inconsistent Sem-MoE vocab/device score shape at %s: %s vs %s.",
                layer_path,
                score_full.shape,
                dp_score_full.shape,
            )
            return None
        dp_score_full += score_full.astype(np.float32, copy=False)
        layers[layer_id] = SemMoeLayerSchedule(
            layer_id=layer_id,
            num_experts=int(expert_labels.shape[0]),
            num_devices=num_devices,
            expert_labels=expert_labels,
            expert_permutation=expert_permutation,
            expert_inverse_permutation=expert_inverse_permutation,
            gate_permutation=np.asarray(gate_permutation, dtype=np.int32),
            gate_inverse_permutation=np.asarray(gate_inverse_permutation, dtype=np.int32),
            score_full=score_full,
            t_full=t_full,
            tp_full=tp_full,
            a_table=a_table,
            ap_table=ap_table,
        )

    if dp_score_full is None:
        logger.warning("Sem-MoE manifest %s does not contain any layers.", manifest_path)
        return None

    return SemMoeSchedule(
        schedule_dir=schedule_dir,
        num_devices=num_devices,
        layer_ids=layer_ids,
        layers=layers,
        dp_score_full=dp_score_full,
        lookback=lookback,
    )


def build_linear_runtime_map(num_experts: int, ep_size: int, ep_rank: int) -> np.ndarray:
    if ep_size <= 0:
        raise ValueError("ep_size must be positive.")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank={ep_rank} is out of range for ep_size={ep_size}.")

    base = num_experts // ep_size
    remainder = num_experts % ep_size
    start = ep_rank * base + min(ep_rank, remainder)
    count = base + (1 if ep_rank < remainder else 0)
    expert_map = np.full((num_experts,), -1, dtype=np.int32)
    if count > 0:
        expert_map[start : start + count] = np.arange(count, dtype=np.int32)
    return expert_map


def build_loader_and_runtime_maps(
    expert_inverse_permutation: np.ndarray,
    ep_size: int,
    ep_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime_map = build_linear_runtime_map(
        num_experts=int(expert_inverse_permutation.shape[0]),
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    loader_map = runtime_map[expert_inverse_permutation]
    return (
        torch.from_numpy(loader_map.copy()),
        torch.from_numpy(runtime_map.copy()),
    )


def bind_sem_moe_model_for_loading(model: torch.nn.Module, model_config: Any) -> None:
    config = sem_moe_config()
    if not config.enabled:
        return
    if getattr(model_config, "quantization", None) is not None:
        logger.warning(
            "Sem-MoE model scheduling currently only supports bf16/unquantized paths; "
            "quantization=%r disables it.",
            model_config.quantization,
        )
        return

    schedule = load_sem_moe_schedule()
    if schedule is None:
        return

    pending_layers: list[SemMoeBoundLayer] = []
    for block in _iter_supported_sparse_moe_blocks(model):
        layer_id = int(getattr(block, "layer_idx"))
        layer_schedule = schedule.layers.get(layer_id)
        if layer_schedule is None:
            logger.warning(
                "Sem-MoE schedule does not contain MoE layer %s required by %s.",
                layer_id,
                type(model).__name__,
            )
            return

        experts = getattr(block, "experts", None)
        gate = getattr(block, "gate", None)
        if experts is None or gate is None or getattr(experts, "_expert_map", None) is None:
            logger.warning(
                "Sem-MoE only supports EP-enabled Qwen sparse MoE blocks; skipping layer %s.",
                layer_id,
            )
            return

        ep_size = int(getattr(experts, "ep_size"))
        ep_rank = int(getattr(experts, "ep_rank"))
        num_experts = int(getattr(block, "n_routed_experts"))
        if schedule.num_devices != ep_size:
            logger.warning(
                "Sem-MoE schedule num_devices=%s does not match EP size=%s; disabling.",
                schedule.num_devices,
                ep_size,
            )
            return
        if layer_schedule.num_experts != num_experts:
            logger.warning(
                "Sem-MoE layer %s expects %s experts but model has %s.",
                layer_id,
                layer_schedule.num_experts,
                num_experts,
            )
            return

        loader_map, runtime_map = build_loader_and_runtime_maps(
            layer_schedule.expert_inverse_permutation,
            ep_size=ep_size,
            ep_rank=ep_rank,
        )
        pending_layers.append(
            SemMoeBoundLayer(
                layer_id=layer_id,
                experts=experts,
                gate=gate,
                schedule=layer_schedule,
                loader_map=loader_map,
                runtime_map=runtime_map,
            )
        )

    if not pending_layers:
        return

    for bound_layer in pending_layers:
        bound_layer.experts.install_sem_moe_loader_map(
            bound_layer.loader_map,
            bound_layer.runtime_map,
        )

    setattr(
        model,
        "_sem_moe_context",
        SemMoeModelContext(schedule=schedule, bound_layers=pending_layers),
    )


def finalize_sem_moe_model(model: torch.nn.Module) -> None:
    context = getattr(model, "_sem_moe_context", None)
    if context is None or context.applied:
        return

    for bound_layer in context.bound_layers:
        # Validate permutation arrays before applying
        _validate_permutation(
            bound_layer.schedule.gate_permutation,
            bound_layer.schedule.num_experts,
            f"layer {bound_layer.layer_id} gate_permutation",
        )
        _validate_permutation(
            bound_layer.schedule.expert_permutation,
            bound_layer.schedule.num_experts,
            f"layer {bound_layer.layer_id} expert_permutation",
        )
        gp = bound_layer.schedule.gate_permutation
        ep = bound_layer.schedule.expert_permutation
        if not np.array_equal(gp, ep):
            logger.warning(
                "Sem-MoE layer %s: gate_permutation differs from expert_permutation! "
                "This may cause incorrect routing. First 10 diffs: gate=%s expert=%s",
                bound_layer.layer_id,
                gp[:10].tolist(),
                ep[:10].tolist(),
            )
        _permute_gate(bound_layer.gate, gp)
        bound_layer.experts.activate_sem_moe_runtime_map()
        local_old_experts = np.nonzero(bound_layer.loader_map.cpu().numpy() >= 0)[0].tolist()
        runtime_map = bound_layer.runtime_map.cpu().numpy()
        local_new_experts = np.nonzero(runtime_map >= 0)[0].tolist()

        logger.info(
            "Applied Sem-MoE placement to layer %s on EP rank %s/%s: "
            "checkpoint experts %s -> runtime experts %s.",
            bound_layer.layer_id,
            getattr(bound_layer.experts, "ep_rank", 0),
            getattr(bound_layer.experts, "ep_size", 1),
            local_old_experts,
            local_new_experts,
        )

    context.applied = True

    # --- TP rebatching setup ---
    config = sem_moe_config()
    if config.mode in {"tp", "both"}:
        from vllm.sem_moe_tp import build_tp_context

        device = next(model.parameters()).device
        tp_ctx = build_tp_context(context.schedule, device)
        if tp_ctx is not None:
            # Attach TP context to the model and all MoE blocks
            model._sem_moe_tp_ctx = tp_ctx  # type: ignore[attr-defined]
            srs_active = not config.debug_fallback
            for block in _iter_supported_sparse_moe_blocks(model):
                block._sem_moe_tp_ctx = tp_ctx  # type: ignore[attr-defined]
                block._sem_moe_srs_active = srs_active  # type: ignore[attr-defined]
            # Also mark decoder layers that contain MoE blocks,
            # and inner model modules that thread input_ids in forward().
            for module in model.modules():
                mlp = getattr(module, "mlp", None)
                if mlp is not None and type(mlp).__name__ in SUPPORTED_BLOCK_TYPES:
                    module._sem_moe_tp_ctx = tp_ctx  # type: ignore[attr-defined]
                    module._sem_moe_srs_active = srs_active  # type: ignore[attr-defined]
                # Set on inner model instances (Qwen3NextModel, Qwen3MoeModel)
                # that read _sem_moe_tp_ctx in forward() to thread input_ids.
                if (
                    hasattr(module, "layers")
                    and hasattr(module, "embed_tokens")
                    and module is not model
                ):
                    module._sem_moe_tp_ctx = tp_ctx  # type: ignore[attr-defined]
            logger.info(
                "Sem-MoE TP rebatching enabled for %d MoE layers (srs=%s).",
                len(tp_ctx.layer_tensors),
                srs_active,
            )

            # Initialize SRS symmetric memory pool when SRS is active
            if srs_active:
                try:
                    from vllm.distributed import get_tp_group
                    from vllm.sem_moe_triton import SemMoeSRSPool

                    tp_group = get_tp_group()
                    srs_pool = SemMoeSRSPool()
                    model_config = getattr(model, "config", None)
                    hidden_size = getattr(model_config, "hidden_size", 2048)
                    max_tokens = int(os.getenv("SEM_MOE_SRS_MAX_TOKENS", "8192"))
                    srs_pool.initialize(
                        max_tokens=max_tokens,
                        hidden_dim=hidden_size,
                        dtype=next(model.parameters()).dtype,
                        num_ranks=tp_group.world_size,
                        group=tp_group.device_group,
                        device=device,
                    )
                    tp_ctx.srs_pool = srs_pool  # type: ignore[attr-defined]
                except Exception:
                    logger.warning(
                        "Failed to initialize SRS symmetric memory pool; "
                        "falling back to NCCL for SRS/SAG.",
                        exc_info=True,
                    )
                    tp_ctx.srs_pool = None  # type: ignore[attr-defined]


class SemMoeDPSelector:
    def __init__(self, score_full: np.ndarray):
        if score_full.ndim != 2:
            raise ValueError("score_full must have shape [vocab_size, num_devices].")
        self.score_full = score_full.astype(np.float32, copy=False)
        self.num_devices = int(score_full.shape[1])
        self.dev_mask = np.ones((self.num_devices,), dtype=bool)

    def pick_rank(self, token_ids: list[int] | tuple[int, ...]) -> int:
        return pick_dp_rank_for_request(
            token_ids=token_ids,
            score_full=self.score_full,
            dev_mask=self.dev_mask,
        )


def make_sem_moe_dp_selector(expected_num_devices: int) -> SemMoeDPSelector | None:
    if not sem_moe_dp_enabled():
        return None
    schedule = load_sem_moe_schedule()
    if schedule is None:
        return None
    if schedule.num_devices != expected_num_devices:
        logger.warning(
            "Sem-MoE DP schedule num_devices=%s does not match runtime DP size=%s; disabling DP routing.",
            schedule.num_devices,
            expected_num_devices,
        )
        return None
    return SemMoeDPSelector(schedule.dp_score_full)


def pick_dp_rank_for_request(
    token_ids: list[int] | tuple[int, ...],
    score_full: np.ndarray,
    dev_mask: np.ndarray,
) -> int:
    token_array = np.asarray(token_ids, dtype=np.int64)
    if token_array.size > 0:
        valid = token_array[(token_array >= 0) & (token_array < score_full.shape[0])]
    else:
        valid = token_array

    if valid.size > 0:
        dev_score = score_full[valid].sum(axis=0, dtype=np.float64)
    else:
        dev_score = np.zeros((score_full.shape[1],), dtype=np.float64)

    masked_score = dev_score.copy()
    masked_score[~dev_mask] = -np.inf
    device_id = int(masked_score.argmax())
    dev_mask[device_id] = False
    if not bool(dev_mask.any()):
        dev_mask[:] = True
    return device_id


def _validate_permutation(perm: np.ndarray, expected_size: int, name: str) -> None:
    if perm.shape != (expected_size,):
        raise ValueError(
            f"Sem-MoE {name}: expected shape ({expected_size},), got {perm.shape}."
        )
    sorted_perm = np.sort(perm)
    expected = np.arange(expected_size, dtype=sorted_perm.dtype)
    if not np.array_equal(sorted_perm, expected):
        raise ValueError(
            f"Sem-MoE {name}: not a valid permutation of [0..{expected_size-1}]. "
            f"min={perm.min()}, max={perm.max()}, unique={len(np.unique(perm))}."
        )


def _iter_supported_sparse_moe_blocks(model: torch.nn.Module):
    for module in model.modules():
        if type(module).__name__ not in SUPPORTED_BLOCK_TYPES:
            continue
        if not hasattr(module, "layer_idx") or not hasattr(module, "experts") or not hasattr(module, "gate"):
            continue
        yield module


def _permute_gate(gate: torch.nn.Module, permutation: np.ndarray) -> None:
    if not hasattr(gate, "weight"):
        raise ValueError(f"Sem-MoE gate permutation requires a weight tensor, got {type(gate)!r}.")
    weight = gate.weight
    perm = torch.as_tensor(permutation, dtype=torch.long, device=weight.device)
    weight.data = weight.data.index_select(0, perm).contiguous()
    bias = getattr(gate, "bias", None)
    if bias is not None:
        bias.data = bias.data.index_select(0, perm).contiguous()


def _load_layer_score_full(
    payload: Any,
    num_devices: int,
    debug_fallback: bool,
) -> np.ndarray:
    if "T_score_full" in payload:
        return payload["T_score_full"].astype(np.float32, copy=True)
    if "T_full" not in payload:
        raise ValueError("Sem-MoE layer artifact is missing T_score_full/T_full.")

    labels = payload["T_full"].astype(np.int64, copy=False)
    if "Tp_full" in payload:
        confidence = payload["Tp_full"].astype(np.float32, copy=False)
    elif debug_fallback:
        confidence = np.ones_like(labels, dtype=np.float32)
    else:
        raise ValueError(
            "Sem-MoE layer artifact is missing T_score_full and Tp_full; "
            "set SEM_MOE_DEBUG_FALLBACK=1 to allow hard one-hot fallback."
        )

    score_full = np.zeros((labels.shape[0], num_devices), dtype=np.float32)
    rows = np.arange(labels.shape[0], dtype=np.int64)
    score_full[rows, labels] = confidence
    return score_full


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_schedule_dir(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if (path / "manifest.json").exists():
        return path
    if (path / "schedule" / "manifest.json").exists():
        return path / "schedule"
    return None


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

ROUTED_EXPERTS_TRACE_DIR_ENV = "VLLM_ROUTED_EXPERTS_TRACE_DIR"


class RoutedExpertsFileExporter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._device_buffer: torch.Tensor | None = None
        self._batch_index = 0
        self._writer_enabled = False
        self.dp_rank = 0
        self.tp_rank = get_tensor_model_parallel_rank()

    @classmethod
    def create_from_env(cls) -> RoutedExpertsFileExporter | None:
        output_dir = os.environ.get(ROUTED_EXPERTS_TRACE_DIR_ENV)
        if not output_dir:
            return None
        return cls(Path(output_dir).expanduser().resolve())

    def init_buffer(
        self,
        max_num_batched_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        if self._device_buffer is not None:
            raise RuntimeError("Routed experts exporter buffer already initialized.")

        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self._writer_enabled = get_tensor_model_parallel_rank() == 0
        if not self._writer_enabled:
            return

        hf_config = vllm_config.model_config.hf_text_config
        num_layers = hf_config.num_hidden_layers
        num_experts_per_tok = hf_config.num_experts_per_tok
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._device_buffer = torch.zeros(
            (max_num_batched_tokens, num_layers, num_experts_per_tok),
            dtype=torch.int32,
            device=current_platform.device_type,
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if self._device_buffer is None:
            return

        ctx = get_forward_context()
        if ctx.dp_metadata is None:
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:
            token_num_per_dp = ctx.dp_metadata.num_tokens_across_dp_cpu[self.dp_rank]
            cumsum = torch.cumsum(ctx.dp_metadata.num_tokens_across_dp_cpu, dim=0)
            end_loc = cumsum[self.dp_rank]
            start_loc = end_loc - token_num_per_dp

        if layer_id >= self._device_buffer.shape[1]:
            return

        self._device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[
            start_loc:end_loc, :
        ]

    def clear_buffer(self) -> None:
        if self._device_buffer is not None:
            self._device_buffer.zero_()

    def export_batch(
        self,
        request_token_ranges: Sequence[tuple[str, int, int]],
    ) -> Path | None:
        if self._device_buffer is None or not request_token_ranges:
            return None

        max_end = max(end for _, _, end in request_token_ranges)
        data = self._device_buffer[:max_end].cpu()
        records = []
        for request_id, start, end in request_token_ranges:
            public_request_id = request_id.split("-", 1)[0]
            records.append(
                {
                    "request_id": public_request_id,
                    "worker_request_id": request_id,
                    "routed_experts": data[start:end].clone(),
                }
            )

        output_path = self.output_dir / (
            f"routed_experts_dp{self.dp_rank:02d}_tp{self.tp_rank:02d}_"
            f"batch{self._batch_index:08d}.pt"
        )
        temp_path = output_path.with_suffix(".tmp")
        torch.save(
            {
                "metadata": {
                    "dp_rank": self.dp_rank,
                    "tp_rank": self.tp_rank,
                    "batch_index": self._batch_index,
                },
                "records": records,
            },
            temp_path,
        )
        temp_path.replace(output_path)
        self._batch_index += 1
        return output_path

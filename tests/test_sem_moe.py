from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.sem_moe import (
    bind_sem_moe_model_for_loading,
    build_loader_and_runtime_maps,
    clear_sem_moe_caches,
    finalize_sem_moe_model,
    pick_dp_rank_for_request,
)
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import DPLBAsyncMPClient


class DummyExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ep_size = 2
        self.ep_rank = 0
        self.register_buffer("_expert_map", torch.tensor([0, 1, -1, -1], dtype=torch.int32))
        self.loader_map: torch.Tensor | None = None
        self.runtime_map: torch.Tensor | None = None
        self.runtime_map_activated = False

    def install_sem_moe_loader_map(
        self,
        loader_map: torch.Tensor,
        runtime_map: torch.Tensor,
    ) -> None:
        self.loader_map = loader_map.clone()
        self.runtime_map = runtime_map.clone()
        self._buffers["_expert_map"] = loader_map.clone()

    def activate_sem_moe_runtime_map(self) -> None:
        assert self.runtime_map is not None
        self.runtime_map_activated = True
        self._buffers["_expert_map"] = self.runtime_map.clone()


class Qwen3NextSparseMoeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 5
        self.n_routed_experts = 4
        self.gate = torch.nn.Linear(3, 4, bias=False)
        self.gate.weight.data.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        self.experts = DummyExperts()


class DummyMoEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Qwen3NextSparseMoeBlock()


class StubSelector:
    def __init__(self, rank: int):
        self.rank = rank
        self.calls: list[list[int]] = []

    def pick_rank(self, token_ids: list[int]) -> int:
        self.calls.append(list(token_ids))
        return self.rank


def _write_schedule(schedule_dir: Path) -> None:
    schedule_dir.mkdir(parents=True, exist_ok=True)
    (schedule_dir / "manifest.json").write_text(
        json.dumps(
            {
                "num_devices": 2,
                "moe_layer_ids": [5],
                "model_name": "dummy/model",
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        schedule_dir / "semmoe_layer5.npz",
        E=np.array([1, 0, 1, 0], dtype=np.int32),
        expert_permutation=np.array([1, 3, 0, 2], dtype=np.int32),
        expert_inverse_permutation=np.array([2, 0, 3, 1], dtype=np.int32),
        gating_column_permutation=np.array([1, 3, 0, 2], dtype=np.int32),
        gating_column_inverse_permutation=np.array([2, 0, 3, 1], dtype=np.int32),
        T_score_full=np.zeros((16, 2), dtype=np.float32),
    )


def test_build_loader_and_runtime_maps_match_permutation() -> None:
    loader_map, runtime_map = build_loader_and_runtime_maps(
        np.array([2, 0, 3, 1], dtype=np.int32),
        ep_size=2,
        ep_rank=0,
    )
    assert runtime_map.tolist() == [0, 1, -1, -1]
    assert loader_map.tolist() == [-1, 0, -1, 1]


def test_pick_dp_rank_for_request_rotates_mask() -> None:
    score_full = np.array(
        [
            [10.0, 1.0],
            [9.0, 2.0],
            [1.0, 10.0],
        ],
        dtype=np.float32,
    )
    dev_mask = np.ones((2,), dtype=bool)
    first = pick_dp_rank_for_request([0, 1], score_full=score_full, dev_mask=dev_mask)
    second = pick_dp_rank_for_request([0, 1], score_full=score_full, dev_mask=dev_mask)
    third = pick_dp_rank_for_request([2], score_full=score_full, dev_mask=dev_mask)
    assert [first, second, third] == [0, 1, 1]
    assert dev_mask.tolist() == [True, False]


def test_bind_and_finalize_sem_moe_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    schedule_dir = tmp_path / "schedule"
    _write_schedule(schedule_dir)
    clear_sem_moe_caches()
    monkeypatch.setenv("SEM_MOE", "1")
    monkeypatch.setenv("SEM_MOE_TABLES", str(schedule_dir))
    model = DummyMoEModel()

    bind_sem_moe_model_for_loading(model, SimpleNamespace(quantization=None))
    assert model.block.experts.loader_map is not None
    assert model.block.experts.loader_map.tolist() == [-1, 0, -1, 1]

    finalize_sem_moe_model(model)

    assert model.block.experts.runtime_map_activated
    assert model.block.experts._expert_map.tolist() == [0, 1, -1, -1]
    assert model.block.gate.weight[:, 0].tolist() == [2.0, 4.0, 1.0, 3.0]


def test_dplb_client_prefers_sem_moe_selector() -> None:
    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = [b"\x00\x00", b"\x00\x01"]
    client.reqs_in_flight = {}
    client.lb_engines = [[0, 0], [0, 0]]
    client.client_count = 1
    client.eng_start_index = 0
    client.sem_moe_dp_selector = StubSelector(rank=1)
    client._sem_moe_selector_num_engines = 2
    client._refresh_sem_moe_dp_selector = lambda: None

    request = EngineCoreRequest(
        request_id="req-0",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    chosen_engine = DPLBAsyncMPClient.get_core_engine_for_request(client, request)

    assert chosen_engine == client.core_engines[1]
    assert client.reqs_in_flight[request.request_id] == client.core_engines[1]
    assert client.sem_moe_dp_selector.calls == [[1, 2, 3]]

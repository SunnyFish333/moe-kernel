"""CPU tests of vLLM plugin dispatch; no checkpoint or CUDA device required."""

from enum import Enum, auto
from types import SimpleNamespace

import pytest

from integrations.vllm import adapter
from moe_kernel import MoEConfig


class Order(Enum):
    NO_OVERLAP = auto()
    MK_INTERNAL_OVERLAPPED = auto()
    MULTI_STREAM_OVERLAPPED = auto()


@pytest.mark.parametrize("mode", ["experts", "full"])
@pytest.mark.parametrize("shared_order", list(Order) + [None])
def test_replaces_routed_path_and_retains_shared_expert(monkeypatch, mode, shared_order):
    calls = []
    config = MoEConfig(num_experts=4, top_k=2, renormalize=False)
    ws = object()
    layer = SimpleNamespace(w13_weight="gate/up", w2_weight="down")

    class Runner:
        def __init__(self):
            self.router = SimpleNamespace(select_experts=self.route)
            self.shared_experts = None if shared_order is None else SimpleNamespace(output=None)

        def route(self, **kwargs):
            assert kwargs == dict(hidden_states="x", router_logits="logits", input_ids="ids")
            calls.append("route")
            return "weights", "experts"

        def _maybe_apply_shared_experts(self, shared_x, order):
            assert shared_x == "shared_x"
            if order is shared_order:
                assert self.shared_experts.output is None
                calls.append("shared")
                self.shared_experts.output = "shared_result"

    def experts(*args, workspace):
        assert args == ("x", "gate/up", "down", "weights", "experts")
        assert workspace is ws
        calls.append("routed")
        return "routed_result"

    def moe(*args, workspace, top_k, renormalize):
        assert args == ("x", "logits", "gate/up", "down")
        assert workspace is ws and top_k == 2 and not renormalize
        calls.append("routed")
        return "routed_result"

    monkeypatch.setattr(adapter, "experts", experts)
    monkeypatch.setattr(adapter, "moe", moe)
    result = adapter._run_routed(
        Runner(), layer, "x", "logits", "shared_x", "ids",
        mode=mode, workspace=ws, config=config, order=Order,
    )
    assert result == (None if shared_order is None else "shared_result", "routed_result")
    expected = ["route", "routed"] if mode == "experts" else ["routed"]
    if shared_order is Order.NO_OVERLAP:
        expected.insert(0, "shared")
    elif shared_order is not None:
        expected.append("shared")
    assert calls == expected


def test_native_does_not_access_vllm_installation(monkeypatch):
    monkeypatch.setenv("MOE_KERNEL_MODE", "native")
    monkeypatch.setattr(adapter, "version", lambda *_: pytest.fail("native must not inspect vLLM"))
    adapter.register()


def test_custom_mode_rejects_different_vllm_version(monkeypatch):
    monkeypatch.setenv("MOE_KERNEL_MODE", "experts")
    monkeypatch.setattr(adapter, "version", lambda *_: "0.20.1")
    with pytest.raises(RuntimeError, match="requires vLLM 0.20.0"):
        adapter.register()


def test_installation_rejects_changed_runner_signature():
    class Runner:
        def _apply_quant_method(self, layer):
            pass

    with pytest.raises(RuntimeError, match="signature"):
        adapter._install(
            Runner, mode="full", max_tokens=32, method_type=None, triton_backend=None,
            router_type=None, order=None, get_config=None,
        )

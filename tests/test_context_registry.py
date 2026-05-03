from __future__ import annotations

import numpy as np
import pytest

from packages.galaxy_agent.context_registry import ContextRegistry


def test_put_and_get_roundtrip() -> None:
    reg = ContextRegistry("req-001")
    arr = np.zeros((10, 10), dtype=np.float32)
    reg.put("image:req-001", arr)
    result = reg.get("image:req-001")
    assert result is arr


def test_get_missing_handle_raises_key_error() -> None:
    reg = ContextRegistry("req-001")
    with pytest.raises(KeyError, match="Handle 'missing' not found"):
        reg.get("missing")


def test_clear_removes_all_entries() -> None:
    reg = ContextRegistry("req-001")
    reg.put("image:req-001", np.zeros((2, 2)))
    reg.put("seg:req-001", object())
    reg.clear()
    with pytest.raises(KeyError):
        reg.get("image:req-001")


def test_standard_handle_names() -> None:
    reg = ContextRegistry("req-abc")
    assert reg.image_handle() == "image:req-abc"
    assert reg.seg_handle() == "seg:req-abc"
    assert reg.metrics_handle() == "metrics:req-abc"


def test_overwrite_handle() -> None:
    reg = ContextRegistry("req-001")
    obj1 = object()
    obj2 = object()
    reg.put("h", obj1)
    reg.put("h", obj2)
    assert reg.get("h") is obj2


def test_request_id_property() -> None:
    reg = ContextRegistry("req-xyz")
    assert reg.request_id == "req-xyz"

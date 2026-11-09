"""Fallbacks when the CDS is down (incident of 2026-10-28): NED and IDIA in the resolver,
SIMBAD mirrors for object metadata, and services skipped for a while after a failure."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
import requests

from packages.galaxy_core.domain.errors import TargetResolutionUnavailableError
from packages.galaxy_core.infrastructure import availability
from packages.galaxy_core.infrastructure import sesame_client as resolver
from packages.galaxy_core.infrastructure.simbad_client import query_object

CLIENT = httpx.AsyncClient
MIRROR = "Coordinates(ICRS,ep=J2000,eq=2000): 00 42 44.330  +41 16 07.50 (IR) C\n"
NED = json.dumps(
    {"Preferred": {"Name": "NGC 2403", "Position": {"RA": 114.213742, "Dec": 65.602684}}}
)


def transport(monkeypatch, handler):
    calls: list[str] = []

    def record(request):
        calls.append(request.url.host)
        return handler(request)

    monkeypatch.setattr(
        resolver.httpx,
        "AsyncClient",
        lambda **kw: CLIENT(transport=httpx.MockTransport(record), **kw),
    )
    return calls


def down(request):
    raise httpx.ConnectTimeout("down")


# ── Name resolver ────────────────────────────────────────────────────


def test_idia_mirror_when_cds_and_harvard_are_down(monkeypatch):
    calls = transport(
        monkeypatch,
        lambda r: (
            httpx.Response(200, text=MIRROR) if r.url.host == "simbad.idia.ac.za" else down(r)
        ),
    )
    assert resolver.resolve("M31") == pytest.approx((10.6847083333333, 41.26875))
    assert calls == ["cds.unistra.fr", "simbad.cfa.harvard.edu", "simbad.idia.ac.za"]


def test_ned_as_last_resort(monkeypatch):
    calls = transport(
        monkeypatch,
        lambda r: (
            httpx.Response(200, text=NED) if r.url.host == "ned.ipac.caltech.edu" else down(r)
        ),
    )
    assert resolver.resolve("NGC 2403") == pytest.approx((114.213742, 65.602684))
    assert calls[-1] == "ned.ipac.caltech.edu"


def test_ned_negative_or_bad_answer_is_unavailable_not_invented(monkeypatch):
    transport(
        monkeypatch,
        lambda r: (
            httpx.Response(200, text='{"ResultCode": 0}')
            if r.url.host == "ned.ipac.caltech.edu"
            else down(r)
        ),
    )
    with pytest.raises(TargetResolutionUnavailableError):
        resolver.resolve("Xyzzy")


def test_down_service_is_skipped_on_the_next_request(monkeypatch):
    calls = transport(
        monkeypatch,
        lambda r: (
            httpx.Response(200, text=MIRROR) if r.url.host == "simbad.cfa.harvard.edu" else down(r)
        ),
    )
    resolver.resolve("M31")
    assert calls == ["cds.unistra.fr", "simbad.cfa.harvard.edu"]
    calls.clear()
    resolver.resolve("M31")
    assert calls == ["simbad.cfa.harvard.edu"], (
        "the CDS must not be asked again during the cool-down"
    )


def test_service_is_tried_again_after_the_cool_down(monkeypatch):
    monkeypatch.setenv("SERVICE_COOL_DOWN_SECONDS", "0")
    calls = transport(
        monkeypatch,
        lambda r: (
            httpx.Response(200, text=MIRROR) if r.url.host == "simbad.cfa.harvard.edu" else down(r)
        ),
    )
    resolver.resolve("M31")
    calls.clear()
    resolver.resolve("M31")
    assert calls[0] == "cds.unistra.fr"


# ── SIMBAD metadata ──────────────────────────────────────────────────


def _tap_response(main_id: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "metadata": [{"name": "main_id"}, {"name": "otype"}],
        "data": [[main_id, "Sy2"]],
    }
    return resp


@patch("packages.galaxy_core.infrastructure.simbad_client.requests.get")
def test_simbad_metadata_from_harvard_when_cds_is_down(mock_get: MagicMock) -> None:
    def fake_get(url, **kwargs):
        if "cds.unistra.fr" in url:
            raise requests.ConnectTimeout("down")
        return _tap_response("M  51")

    mock_get.side_effect = fake_get
    record = query_object("M51")
    assert record == {"main_id": "M  51", "otype": "Sy2"}
    assert [c.args[0] for c in mock_get.call_args_list] == [
        "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
        "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync",
    ]
    # short timeouts: (connect, read), not 15 s
    assert mock_get.call_args_list[0].kwargs["timeout"] == (3, 6)
    # the next request skips the CDS
    mock_get.reset_mock()
    query_object("M51")
    assert [c.args[0] for c in mock_get.call_args_list] == [
        "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync"
    ]


@patch("packages.galaxy_core.infrastructure.simbad_client.requests.get")
def test_simbad_metadata_none_when_every_service_is_down(mock_get: MagicMock) -> None:
    mock_get.side_effect = requests.ConnectionError("down")
    assert query_object("M51") is None
    assert mock_get.call_count == 3
    assert not availability.is_available("simbad_cds")

from __future__ import annotations

import io
import urllib.parse
from typing import Any

import pytest
from PIL import Image

from packages.galaxy_core.application.resolve_and_fetch_service import resolve_and_fetch
from packages.galaxy_core.infrastructure.dss2_client import (
    PLATE_BLUE,
    PLATE_IR,
    PLATE_RED,
    fetch_cutout_jpeg,
    get_image_url,
    survey_to_plate,
)


def test_survey_to_plate_supports_common_aliases() -> None:
    assert survey_to_plate("DSS") == PLATE_RED
    assert survey_to_plate("dss2") == PLATE_RED
    assert survey_to_plate("DSS2-red") == PLATE_RED
    assert survey_to_plate("DSS2 blue") == PLATE_BLUE
    assert survey_to_plate("POSS2UKSTU_IR") == PLATE_IR
    assert survey_to_plate("SDSS") is None


def test_get_image_url_builds_dss2_query() -> None:
    url = get_image_url(
        ra_deg=370.1234567,
        dec_deg=95.0,
        size_arcmin=0.05,
        plate=PLATE_BLUE,
    )
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)

    assert parsed.netloc == "archive.stsci.edu"
    assert parsed.path == "/cgi-bin/dss_search"
    assert params["v"] == [PLATE_BLUE]
    assert params["r"] == ["10.123457"]
    assert params["d"] == ["90.0"]
    assert params["h"] == ["0.1"]
    assert params["w"] == ["0.1"]
    assert params["f"] == ["gif"]


def test_fetch_cutout_jpeg_converts_from_gif(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("L", (8, 8), color=120)
    gif_stream = io.BytesIO()
    image.save(gif_stream, format="GIF")
    gif_bytes = gif_stream.getvalue()

    captured: dict[str, Any] = {}

    class Response:
        headers = {"Content-Type": "image/gif"}

        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: float, **kwargs: Any) -> Response:
        captured["url"] = url
        captured["timeout"] = timeout
        return Response(gif_bytes)

    monkeypatch.setattr(
        "packages.galaxy_core.infrastructure.dss2_client.requests.get",
        fake_get,
    )

    jpeg_bytes = fetch_cutout_jpeg(
        ra_deg=10.0,
        dec_deg=20.0,
        size_arcmin=8.0,
    )

    assert jpeg_bytes.startswith(b"\xff\xd8\xff")
    assert "archive.stsci.edu/cgi-bin/dss_search" in str(captured["url"])
    assert captured["timeout"] == 60.0


def test_fetch_cutout_jpeg_fails_if_response_not_gif(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        headers = {"Content-Type": "image/x-fits"}
        content = b"SIMPLE  = T"

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "packages.galaxy_core.infrastructure.dss2_client.requests.get",
        lambda url, timeout, **kwargs: Response(),
    )

    with pytest.raises(ValueError, match="did not return a GIF image"):
        fetch_cutout_jpeg(
            ra_deg=10.0,
            dec_deg=20.0,
            size_arcmin=8.0,
        )


def test_resolve_and_fetch_uses_dss2_client_for_dss_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"provider": 0}

    monkeypatch.setattr(
        "packages.galaxy_core.application.resolve_and_fetch_service.sesame_resolve",
        lambda name: (10.0, 20.0),
    )

    def fake_resolve_image_url_for_survey(**kwargs: Any) -> str:
        calls["provider"] += 1
        assert kwargs["survey"] == "DSS"
        assert kwargs["ra_deg"] == 10.0
        assert kwargs["dec_deg"] == 20.0
        return "https://archive.stsci.edu/cgi-bin/dss_search?v=poss2ukstu_red"

    monkeypatch.setattr(
        "packages.galaxy_core.application.resolve_and_fetch_service.resolve_image_url_for_survey",
        fake_resolve_image_url_for_survey,
    )

    resolved = resolve_and_fetch(name="M31", catalog="DSS")

    assert resolved.survey_used == "DSS"
    assert resolved.image_url.startswith("https://archive.stsci.edu/cgi-bin/dss_search")
    assert calls["provider"] == 1

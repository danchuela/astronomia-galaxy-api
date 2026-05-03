"""Tests for SIMBAD TAP client and MAST HST/JWST search."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from packages.galaxy_core.infrastructure.mast_hst_client import (
    format_hst_jwst_info,
    search_hst_jwst,
)
from packages.galaxy_core.infrastructure.simbad_client import (
    _escape_adql,
    format_object_info,
    query_object,
)

# ---------------------------------------------------------------------------
# SIMBAD: query_object
# ---------------------------------------------------------------------------


def _mock_simbad_response(data: list[list[Any]], columns: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "metadata": [{"name": c} for c in columns],
        "data": data,
    }
    return resp


class TestSimbadQueryObject:
    COLUMNS = [
        "main_id",
        "otype",
        "otype_long",
        "morph_type",
        "rvz_radvel",
        "rvz_redshift",
        "sp_type",
    ]

    @patch("packages.galaxy_core.infrastructure.simbad_client.requests.get")
    def test_returns_record_for_known_object(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_simbad_response(
            data=[["M  51", "Sy2", "Seyfert 2", "SA(s)bc", 463.0, 0.00154, None]],
            columns=self.COLUMNS,
        )
        result = query_object("M51")
        assert result is not None
        assert result["main_id"] == "M  51"
        assert result["otype"] == "Sy2"
        assert result["morph_type"] == "SA(s)bc"
        assert result["rvz_radvel"] == 463.0

    @patch("packages.galaxy_core.infrastructure.simbad_client.requests.get")
    def test_returns_none_for_unknown_object(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_simbad_response(data=[], columns=self.COLUMNS)
        result = query_object("FAKEGALAXYZ99")
        assert result is None

    def test_returns_none_for_empty_name(self) -> None:
        result = query_object("")
        assert result is None
        result = query_object("   ")
        assert result is None

    @patch("packages.galaxy_core.infrastructure.simbad_client.requests.get")
    def test_handles_network_error_gracefully(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = ConnectionError("DNS failure")
        result = query_object("M51")
        assert result is None

    @patch("packages.galaxy_core.infrastructure.simbad_client.requests.get")
    def test_skips_none_values_in_record(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_simbad_response(
            data=[["NGC 1300", "G", "Galaxy", None, None, None, None]],
            columns=self.COLUMNS,
        )
        result = query_object("NGC 1300")
        assert result is not None
        assert "morph_type" not in result
        assert "rvz_radvel" not in result


class TestSimbadFormatObjectInfo:
    def test_formats_full_record(self) -> None:
        record = {
            "main_id": "M  51",
            "otype_txt": "Seyfert 2",
            "morph_type": "SA(s)bc",
            "rvz_radvel": 463.0,
            "rvz_redshift": 0.001544,
        }
        text = format_object_info(record)
        assert "M  51" in text
        assert "Seyfert 2" in text
        assert "SA(s)bc" in text
        assert "463" in text
        assert "0.001544" in text

    def test_formats_minimal_record(self) -> None:
        record = {"main_id": "NGC 1300", "otype": "G"}
        text = format_object_info(record)
        assert "NGC 1300" in text
        assert "G" in text


class TestEscapeAdql:
    def test_escapes_single_quotes(self) -> None:
        assert _escape_adql("O'Brien") == "O''Brien"

    def test_no_quotes_unchanged(self) -> None:
        assert _escape_adql("M51") == "M51"


# ---------------------------------------------------------------------------
# MAST HST/JWST: search_hst_jwst
# ---------------------------------------------------------------------------


def _mock_mast_response(data: list[dict[str, Any]]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.content = json.dumps({"data": data}).encode("utf-8")
    return resp


class TestMastHstSearch:
    @patch("packages.galaxy_core.infrastructure.mast_hst_client.requests.post")
    def test_returns_hst_observation_with_jpeg(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _mock_mast_response(
            [
                {
                    "obs_collection": "HST",
                    "obs_id": "hst_12345",
                    "instrument_name": "ACS/WFC",
                    "filters": "F814W",
                    "jpegURL": "https://mast.example.com/preview.jpg",
                    "dataproduct_type": "image",
                }
            ]
        )
        result = search_hst_jwst(189.99, 12.39)
        assert result is not None
        assert result["collection"] == "HST"
        assert result["jpeg_url"] == "https://mast.example.com/preview.jpg"
        assert result["instrument"] == "ACS/WFC"

    @patch("packages.galaxy_core.infrastructure.mast_hst_client.requests.post")
    def test_returns_none_when_no_observations(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _mock_mast_response([])
        result = search_hst_jwst(0.0, 0.0)
        assert result is None

    @patch("packages.galaxy_core.infrastructure.mast_hst_client.requests.post")
    def test_handles_network_error_gracefully(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = ConnectionError("timeout")
        result = search_hst_jwst(189.99, 12.39)
        assert result is None

    @patch("packages.galaxy_core.infrastructure.mast_hst_client.requests.post")
    def test_prefers_hst_over_jwst(self, mock_post: MagicMock) -> None:
        # First call (HST) returns data, second (JWST) should not be called
        mock_post.return_value = _mock_mast_response(
            [
                {
                    "obs_collection": "HST",
                    "obs_id": "hst_99",
                    "instrument_name": "WFC3",
                    "filters": "F555W",
                    "jpegURL": "https://mast.example.com/hst.jpg",
                    "dataproduct_type": "image",
                }
            ]
        )
        result = search_hst_jwst(189.99, 12.39)
        assert result is not None
        assert result["collection"] == "HST"
        assert mock_post.call_count == 1  # Only HST queried


class TestMastFormatHstJwstInfo:
    def test_formats_hst_observation(self) -> None:
        record = {"collection": "HST", "instrument": "ACS/WFC", "filters": "F814W"}
        text = format_hst_jwst_info(record)
        assert "HST" in text
        assert "ACS/WFC" in text
        assert "F814W" in text

    def test_formats_jwst_observation(self) -> None:
        record = {"collection": "JWST", "instrument": "NIRCam", "filters": "F200W"}
        text = format_hst_jwst_info(record)
        assert "JWST" in text
        assert "NIRCam" in text

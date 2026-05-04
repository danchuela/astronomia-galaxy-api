from __future__ import annotations

import argparse
import json
import re
import socket
import ssl
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

HIPS2FITS_BASE = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
SKYVIEW_RUNQUERY_URL = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"

DEFAULT_RA = 189.99763275
DEFAULT_DEC = -11.62305449
DEFAULT_PIXELS = 300
TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    details: str


def _https_dns_tls(hostname: str, port: int = 443) -> ProbeResult:
    started = time.time()
    try:
        addr = socket.gethostbyname(hostname)
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=TIMEOUT_SECONDS) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as tls_sock:
                cert = tls_sock.getpeercert()
        elapsed_ms = int((time.time() - started) * 1000)
        subject = cert.get("subject", ()) if cert else ()
        return ProbeResult(
            name=f"DNS/TLS {hostname}",
            ok=True,
            details=f"ip={addr}, elapsed_ms={elapsed_ms}, cert_subject={subject}",
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - started) * 1000)
        return ProbeResult(
            name=f"DNS/TLS {hostname}",
            ok=False,
            details=f"elapsed_ms={elapsed_ms}, error={type(exc).__name__}: {exc}",
        )


def _http_get(url: str, name: str) -> ProbeResult:
    started = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "provider-probe/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
        elapsed_ms = int((time.time() - started) * 1000)
        body_preview = body[:200].decode("utf-8", errors="replace").replace("\n", " ")
        ok = 200 <= status < 300
        return ProbeResult(
            name=name,
            ok=ok,
            details=(
                f"status={status}, elapsed_ms={elapsed_ms}, content_type={content_type}, "
                f"bytes={len(body)}, body_preview={body_preview!r}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - started) * 1000)
        return ProbeResult(
            name=name,
            ok=False,
            details=f"elapsed_ms={elapsed_ms}, error={type(exc).__name__}: {exc}",
        )


def _build_hips_url(hips_id: str, ra: float, dec: float, fov_deg: float, pixels: int) -> str:
    params = {
        "hips": hips_id,
        "width": pixels,
        "height": pixels,
        "projection": "SIN",
        "fov": round(fov_deg, 6),
        "ra": round(ra, 6),
        "dec": round(dec, 6),
        "format": "jpg",
        "coordsys": "icrs",
    }
    return f"{HIPS2FITS_BASE}?{urllib.parse.urlencode(params)}"


def _probe_skyview(survey: str, ra: float, dec: float, pixels: int, fov_deg: float) -> ProbeResult:
    started = time.time()
    scale_arcsec = (fov_deg * 3600.0) / pixels
    data = urllib.parse.urlencode(
        {
            "Position": f"{ra},{dec}",
            "Survey": survey,
            "Pixels": str(pixels),
            "Scale": str(round(scale_arcsec, 4)),
            "Coordinates": "J2000",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        SKYVIEW_RUNQUERY_URL,
        data=data,
        headers={"User-Agent": "provider-probe/1.0"},
        method="POST",
    )

    url_pattern = re.compile(
        r'href="(https?://[^"]*skyview[^"]*\.nasa\.gov[^"]+\.(?:fits|fits\.gz|jpg|jpeg|png))"',
        re.IGNORECASE,
    )
    rel_pattern = re.compile(r'href="(/[^"]+\.(?:fits|fits\.gz|jpg|jpeg|png))"', re.IGNORECASE)
    img_pattern = re.compile(
        r'src="(https?://[^"]*skyview[^"]*\.nasa\.gov[^"]+\.(?:jpg|jpeg|png))"',
        re.IGNORECASE,
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
        elapsed_ms = int((time.time() - started) * 1000)
        m1 = url_pattern.search(body)
        m2 = rel_pattern.search(body)
        m3 = img_pattern.search(body)
        found = (
            (m1 and m1.group(1))
            or (m2 and f"https://skyview.gsfc.nasa.gov{m2.group(1)}")
            or (m3 and m3.group(1))
        )
        ok = status == 200 and bool(found)
        details = f"status={status}, elapsed_ms={elapsed_ms}, found_url={found!r}"
        if not found:
            details += f", html_preview={body[:240].replace(chr(10), ' ')!r}"
        return ProbeResult(name=f"SkyView {survey}", ok=ok, details=details)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - started) * 1000)
        return ProbeResult(
            name=f"SkyView {survey}",
            ok=False,
            details=f"elapsed_ms={elapsed_ms}, error={type(exc).__name__}: {exc}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostico independiente de proveedores astronomicos."
    )
    parser.add_argument("--ra", type=float, default=DEFAULT_RA)
    parser.add_argument("--dec", type=float, default=DEFAULT_DEC)
    parser.add_argument("--fov-arcmin", type=float, default=10.0)
    parser.add_argument("--pixels", type=int, default=DEFAULT_PIXELS)
    args = parser.parse_args()

    fov_deg = args.fov_arcmin / 60.0
    hips_urls = {
        "HiPS GALEX": _build_hips_url("GALEX", args.ra, args.dec, fov_deg, args.pixels),
        "HiPS 2MASS-J": _build_hips_url("CDS/P/2MASS/J", args.ra, args.dec, fov_deg, args.pixels),
    }

    results = [
        _https_dns_tls("alasky.cds.unistra.fr"),
        _https_dns_tls("skyview.gsfc.nasa.gov"),
    ]
    for name, url in hips_urls.items():
        results.append(_http_get(url, name))
    results.append(_probe_skyview("GALEX", args.ra, args.dec, args.pixels, fov_deg))
    results.append(_probe_skyview("2MASS-J", args.ra, args.dec, args.pixels, fov_deg))

    print(json.dumps([r.__dict__ for r in results], indent=2, ensure_ascii=False))

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\nFAILED_CHECKS={len(failed)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

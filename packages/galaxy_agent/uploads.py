"""Bounded, content-validated user images. Preview pixels never replace FITS data."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from PIL import Image, ImageOps
from PIL.TiffImagePlugin import TiffImageFile
from pydantic import BaseModel, Field, field_validator

from packages.galaxy_agent.image_selection import ImageRegion, SourcePoint

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_PIXELS = 2048 * 2048
MAX_ENCODED_LENGTH = 4 * ((MAX_UPLOAD_BYTES + 2) // 3) + 100
EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "fits", "fit", "fts"}


class UploadedImage(BaseModel):
    filename: str = Field(min_length=1, max_length=180)
    data: str = Field(min_length=1, max_length=MAX_ENCODED_LENGTH, repr=False)
    hdu_index: int | None = Field(default=None, ge=0, le=63)
    psf_hdu_index: int | None = Field(default=None, ge=0, le=63)
    error_hdu_index: int | None = Field(default=None, ge=0, le=63)
    error_kind: Literal["sigma", "variance", "inverse_variance"] = "sigma"
    source_point: SourcePoint | None = None
    region: ImageRegion | None = None
    uncertainty_samples: Literal[0, 16] = 0

    @field_validator("filename")
    @classmethod
    def safe_filename(cls, value: str) -> str:
        value = value.replace("\\", "/").rsplit("/", 1)[-1]
        if any(ord(c) < 32 for c in value) or value.rsplit(".", 1)[-1].lower() not in EXTENSIONS:
            raise ValueError("Supported files: PNG, JPEG, TIFF and 2D FITS")
        return value


class UploadedImageError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)

    def user_message(self, lang: str) -> str:
        messages = {
            "invalid": (
                "No se pudo leer el archivo. Adjunta una imagen PNG, JPEG, TIFF o FITS 2D válida.",
                "The file could not be read. Attach a valid PNG, JPEG, TIFF or 2D FITS image.",
            ),
            "size": (
                "El archivo supera los 8 MiB permitidos.",
                "The file exceeds the 8 MiB limit.",
            ),
            "pixels": (
                "La imagen debe tener al menos 16 píxeles por lado y un máximo de 4 megapíxeles. "
                "No la he reducido automáticamente.",
                "The image must be at least 16 pixels per side and at most 4 megapixels. "
                "I have not resized it automatically.",
            ),
            "planes": (
                "El archivo contiene varios planos. Selecciona el índice HDU de la imagen FITS "
                "o exporta una sola imagen 2D.",
                "The file contains multiple planes. Select the FITS image HDU index "
                "or export a single 2D image.",
            ),
            "hdu": (
                "El HDU seleccionado no contiene una imagen 2D válida.",
                "The selected HDU does not contain a valid 2D image.",
            ),
            "ancillary": (
                "Revisa los HDU auxiliares: el mapa de error debe coincidir con la imagen; "
                "la PSF debe ser finita, positiva, centrada y de tamaño impar (máximo 129 × 129).",
                "Check the auxiliary HDUs: the error map must match the image; "
                "the PSF must be finite, nonnegative, centered and odd-sized (at most 129 × 129).",
            ),
            "uncertainty_size": (
                "La estimación de estabilidad admite imágenes de hasta 512 × 512 píxeles. "
                "Desactívala para analizar esta imagen sin reducirla.",
                "Stability estimation supports images up to 512 × 512 pixels. "
                "Disable it to analyse this image without resizing it.",
            ),
        }
        text = messages.get(self.code, messages["invalid"])[lang == "en"]
        return text + (" " + self.detail if self.code == "planes" else "")


@dataclass
class PreparedUpload:
    pixels: np.ndarray
    original: bytes
    preview: bytes
    extension: str
    provenance: dict[str, Any]
    noise_sigma: np.ndarray | None = None
    psf: np.ndarray | None = None


def _check_shape(shape: tuple[int, ...]) -> None:
    if len(shape) != 2 or min(shape) < 16 or shape[0] * shape[1] > MAX_UPLOAD_PIXELS:
        raise UploadedImageError("pixels")


def _fits_pixels(raw: bytes, index: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    if not raw.startswith(b"SIMPLE  =") or len(raw) % 2880:
        raise UploadedImageError("invalid")
    with fits.open(io.BytesIO(raw), memmap=False, lazy_load_hdus=True) as hdus:
        candidates = []
        for i, hdu in enumerate(hdus):
            if i >= 64:
                raise UploadedImageError("invalid")
            if type(hdu) in (fits.PrimaryHDU, fits.ImageHDU) and hdu.header.get("NAXIS") == 2:
                candidates.append(i)
        if index is None:
            science = [i for i in candidates if hdus[i].name.upper() == "SCI"]
            choices = science or candidates
            if len(choices) != 1:
                detail = "HDU: " + ", ".join(str(i) for i in choices)
                raise UploadedImageError("planes" if choices else "hdu", detail)
            index = choices[0]
        if index not in candidates:
            raise UploadedImageError("hdu")
        hdu = hdus[index]
        shape = (int(hdu.header["NAXIS2"]), int(hdu.header["NAXIS1"]))
        _check_shape(shape)
        # Astropy applies BSCALE/BZERO; retain signed, native intensity values in float64.
        pixels = np.array(hdu.data, dtype=np.float64, copy=True)
        if pixels.shape != shape or not np.isfinite(pixels).any():
            raise UploadedImageError("invalid")
        metadata = {
            key: str(hdu.header[key])[:80]
            for key in ("BUNIT", "FILTER", "INSTRUME", "EXPTIME")
            if key in hdu.header
        }
        return pixels, {
            "input_kind": "fits",
            "hdu_index": index,
            "header": metadata,
            "pixel_transform": "BSCALE/BZERO only; no display stretch or resampling",
        }


def fits_hdus(raw: bytes) -> list[dict[str, Any]]:
    """Read bounded headers only; never decompress or load unrelated planes."""
    with fits.open(io.BytesIO(raw), memmap=False, lazy_load_hdus=True) as hdus:
        result = []
        for i, hdu in enumerate(hdus):
            if i >= 64:
                raise UploadedImageError("invalid")
            if type(hdu) in (fits.PrimaryHDU, fits.ImageHDU) and hdu.header.get("NAXIS") == 2:
                result.append(
                    {
                        "index": i,
                        "name": str(hdu.name)[:40],
                        "width": int(hdu.header["NAXIS1"]),
                        "height": int(hdu.header["NAXIS2"]),
                    }
                )
        return result


def _auxiliary_pixels(raw: bytes, index: int, shape: tuple[int, ...] | None) -> np.ndarray:
    with fits.open(io.BytesIO(raw), memmap=False, lazy_load_hdus=True) as hdus:
        if index >= len(hdus) or type(hdus[index]) not in (fits.PrimaryHDU, fits.ImageHDU):
            raise UploadedImageError("ancillary")
        hdu = hdus[index]
        dims = (int(hdu.header.get("NAXIS2", 0)), int(hdu.header.get("NAXIS1", 0)))
        if hdu.header.get("NAXIS") != 2:
            raise UploadedImageError("ancillary")
        if shape is not None and dims != shape:
            raise UploadedImageError("ancillary")
        if shape is None and (min(dims) < 3 or max(dims) > 129 or any(n % 2 == 0 for n in dims)):
            raise UploadedImageError("ancillary")
        return np.array(hdu.data, dtype=np.float64, copy=True)


def prepare_upload(upload: UploadedImage) -> PreparedUpload:
    try:
        raster_preview: bytes | None = None
        encoded = upload.data
        if encoded.startswith("data:"):
            header, encoded = encoded.split(",", 1)
            if not header.endswith(";base64"):
                raise UploadedImageError("invalid")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise UploadedImageError("size")
        extension = upload.filename.rsplit(".", 1)[-1].lower()
        if extension in {"fits", "fit", "fts"}:
            pixels, info = _fits_pixels(raw, upload.hdu_index)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw), formats=("PNG", "JPEG", "TIFF")) as source:
                    image: Image.Image = source
                    expected = {
                        "png": "PNG",
                        "jpg": "JPEG",
                        "jpeg": "JPEG",
                        "tif": "TIFF",
                        "tiff": "TIFF",
                    }[extension]
                    if image.format != expected:
                        raise UploadedImageError("invalid")
                    _check_shape((image.height, image.width))
                    if isinstance(source, TiffImageFile) and source.tag_v2.get(259, 1) not in {
                        1,
                        5,
                        7,
                        8,
                        32773,
                        32946,
                    }:
                        raise UploadedImageError("invalid")
                    try:
                        image.seek(1)
                    except EOFError:
                        image.seek(0)
                    else:
                        raise UploadedImageError("planes")
                    image = ImageOps.exif_transpose(image)
                    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                        rgba = image.convert("RGBA")
                        if rgba.getchannel("A").getextrema() != (255, 255):
                            raise UploadedImageError("invalid")
                    if image.mode not in ("I", "F", "I;16", "I;16B", "I;16L"):
                        display = Image.fromarray(np.array(image.convert("RGB")))
                        display_buffer = io.BytesIO()
                        display.save(display_buffer, format="PNG")
                        raster_preview = display_buffer.getvalue()
                    if image.mode not in ("I", "F", "I;16", "I;16B", "I;16L"):
                        image = image.convert("L")
                    pixels = np.array(image, dtype=np.float64, copy=True)
                    info = {
                        "input_kind": "rendered",
                        "pixel_transform": (
                            "EXIF orientation; luminance for RGB; native grayscale precision"
                        ),
                    }
        finite = pixels[np.isfinite(pixels)]
        if not finite.size:
            raise UploadedImageError("invalid")
        low, high = np.percentile(finite, [1, 99.5])
        scaled = np.clip(
            (np.nan_to_num(pixels, nan=float(low)) - low) / max(high - low, 1e-12), 0, 1
        )
        preview = Image.fromarray(np.uint8(np.arcsinh(10 * scaled) / np.arcsinh(10) * 255))
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")
        preview_bytes = raster_preview or buffer.getvalue()
        sigma = psf = None
        if upload.uncertainty_samples and max(pixels.shape) > 512:
            raise UploadedImageError("uncertainty_size")
        auxiliary = (upload.psf_hdu_index, upload.error_hdu_index)
        if any(i is not None for i in auxiliary):
            if info["input_kind"] != "fits" or info["hdu_index"] in auxiliary:
                raise UploadedImageError("ancillary")
            if auxiliary[0] is not None and auxiliary[0] == auxiliary[1]:
                raise UploadedImageError("ancillary")
            if upload.psf_hdu_index is not None:
                psf = _auxiliary_pixels(raw, upload.psf_hdu_index, None)
                if not np.isfinite(psf).all() or (psf < 0).any() or psf.sum() <= 0:
                    raise UploadedImageError("ancillary")
                psf /= psf.sum()
                yy, xx = np.indices(psf.shape)
                if (
                    abs(float((xx * psf).sum()) - (psf.shape[1] - 1) / 2) > 1
                    or abs(float((yy * psf).sum()) - (psf.shape[0] - 1) / 2) > 1
                ):
                    raise UploadedImageError("ancillary")
            if upload.error_hdu_index is not None:
                errors = _auxiliary_pixels(raw, upload.error_hdu_index, pixels.shape)
                valid = np.isfinite(errors) & (errors > 0)
                if not valid.any():
                    raise UploadedImageError("ancillary")
                sigma = np.full(pixels.shape, np.nan)
                if upload.error_kind == "sigma":
                    sigma[valid] = errors[valid]
                elif upload.error_kind == "variance":
                    sigma[valid] = np.sqrt(errors[valid])
                else:
                    sigma[valid] = 1 / np.sqrt(errors[valid])
        info.update(
            source="upload",
            filename=upload.filename,
            sha256=hashlib.sha256(raw).hexdigest(),
            preview_sha256=hashlib.sha256(preview_bytes).hexdigest(),
            shape=list(pixels.shape),
            calibrated=False,
            psf="user_supplied" if psf is not None else "not_supplied",
            psf_hdu_index=upload.psf_hdu_index,
            error_hdu_index=upload.error_hdu_index,
            noise_model="user_sigma" if sigma is not None else "background_sigma_only",
            error_kind=upload.error_kind if sigma is not None else None,
            source_point=upload.source_point.model_dump() if upload.source_point else None,
            region=upload.region.model_dump() if upload.region else None,
            uncertainty_samples=upload.uncertainty_samples,
            viewer_capture={},
            coordinates={},
            preview_transform=(
                "RGB pixels, metadata removed"
                if raster_preview
                else "percentile 1/99.5 + asinh, display only"
            ),
        )
        return PreparedUpload(pixels, raw, preview_bytes, extension, info, sigma, psf)
    except UploadedImageError:
        raise
    except (
        ValueError,
        OSError,
        TypeError,
        IndexError,
        KeyError,
        OverflowError,
        binascii.Error,
        VerifyError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise UploadedImageError("invalid") from exc

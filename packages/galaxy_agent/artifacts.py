from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from packages.galaxy_agent.models import Artifact


class ArtifactWriter(Protocol):
    def save_mask(self, request_id: str, mask: np.ndarray) -> Artifact: ...

    def save_measurements(self, request_id: str, payload: dict[str, Any]) -> Artifact: ...

    def save_image(self, request_id: str, image_bytes: bytes) -> Artifact: ...

    def save_plot(self, request_id: str, plot_name: str, png_bytes: bytes) -> Artifact: ...


class ArtifactStore:
    def __init__(self, base_dir: str = "artifacts") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _request_dir(self, request_id: str) -> Path:
        safe_id = request_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        path = self.base_dir / safe_id
        resolved = path.resolve()
        if not resolved.is_relative_to(self.base_dir.resolve()):
            raise ValueError(f"Invalid request_id: {request_id!r}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_mask(self, request_id: str, mask: np.ndarray) -> Artifact:
        path = self._request_dir(request_id) / "mask.png"
        img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        img.save(path)
        return Artifact(type="mask", path=self._relative(path))

    def save_measurements(self, request_id: str, payload: dict[str, Any]) -> Artifact:
        path = self._request_dir(request_id) / "measurements.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return Artifact(type="report", path=self._relative(path))

    def save_image(self, request_id: str, image_bytes: bytes) -> Artifact:
        path = self._request_dir(request_id) / "image.jpg"
        path.write_bytes(image_bytes)
        return Artifact(type="image", path=self._relative(path))

    def save_plot(self, request_id: str, plot_name: str, png_bytes: bytes) -> Artifact:
        safe_name = plot_name.replace("/", "_").replace("..", "_")
        path = self._request_dir(request_id) / f"plot-{safe_name}.png"
        path.write_bytes(png_bytes)
        return Artifact(type="plot", path=self._relative(path))

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return path.as_posix()

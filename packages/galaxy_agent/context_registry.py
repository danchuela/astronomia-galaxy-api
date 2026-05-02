from __future__ import annotations


class ContextRegistry:
    """Request-scoped store for non-JSON-serializable objects (numpy arrays, dataclasses).

    The LangGraph agent passes opaque handle strings between tools; this registry
    maps those handles to the actual in-memory objects for the duration of one request.
    """

    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        self._store: dict[str, object] = {}

    @property
    def request_id(self) -> str:
        return self._request_id

    def put(self, handle: str, value: object) -> None:
        self._store[handle] = value

    def get(self, handle: str) -> object:
        if handle not in self._store:
            raise KeyError(
                f"Handle '{handle}' not found in registry for request '{self._request_id}'"
            )
        return self._store[handle]

    def clear(self) -> None:
        self._store.clear()

    # Convenience helpers for the standard handle naming convention
    def image_handle(self) -> str:
        return f"image:{self._request_id}"

    def seg_handle(self) -> str:
        return f"seg:{self._request_id}"

    def metrics_handle(self) -> str:
        return f"metrics:{self._request_id}"

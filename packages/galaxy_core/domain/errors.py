"""Expected target-resolution outcomes, distinct from service failures."""


class TargetNotFoundError(LookupError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"No position found for '{name}'")


class TargetResolutionUnavailableError(RuntimeError):
    """No trusted resolver could return a position within the lookup budget."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Name resolution temporarily unavailable for '{name}'")


class NoSourceDetectedError(LookupError):
    """The supplied field does not contain a measurable detected source."""


class ImageIntegrityError(ValueError):
    """The viewer capture is missing or no longer matches its field metadata."""


class ImageCapacityError(ValueError):
    """A valid image exceeds the bounded scientific workload budget."""

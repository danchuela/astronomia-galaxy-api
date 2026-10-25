"""Expected target-resolution outcomes, distinct from service failures."""


class TargetNotFoundError(LookupError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"No position found for '{name}'")


class NoSourceDetectedError(LookupError):
    """The supplied field does not contain a measurable detected source."""


class ImageIntegrityError(ValueError):
    """The viewer capture is missing or no longer matches its field metadata."""

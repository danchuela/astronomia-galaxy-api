"""Expected target-resolution outcomes, distinct from service failures."""


class TargetNotFoundError(LookupError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"No position found for '{name}'")

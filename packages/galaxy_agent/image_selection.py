"""Normalized image coordinates, independent of display size and FITS WCS."""

from pydantic import BaseModel, Field, model_validator


class ImageRegion(BaseModel):
    left: float = Field(ge=0, le=1, allow_inf_nan=False)
    top: float = Field(ge=0, le=1, allow_inf_nan=False)
    right: float = Field(ge=0, le=1, allow_inf_nan=False)
    bottom: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def nonempty(self) -> "ImageRegion":
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("Empty image region")
        return self


class SourcePoint(BaseModel):
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)

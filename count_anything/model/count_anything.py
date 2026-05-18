"""Public CountAnything model alias.

`Sam3Image` remains the underlying implementation class so upstream `sam3.pt`
checkpoint keys stay valid.
"""

from sam3.model.sam3_image import Sam3Image

CountAnythingModel = Sam3Image

__all__ = ["CountAnythingModel"]

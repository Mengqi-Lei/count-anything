"""Legacy P2P branch import path.

Public code should import `count_anything.model.pixel_dense_counter` directly.
This module only keeps old internal imports/configs alive.
"""

from count_anything.model.pixel_dense_counter import (  # noqa: F401
    DenseCountingFeatureAdapter,
    DensePointClassificationHead,
    DensePointRegressionHead,
    DenseReferencePoints,
    ForegroundClassificationHead,
    PDCFeatureAdapter,
    PDCResidualAdapterBlock,
    PixelLevelDenseCounter,
    PointOffsetHead,
    generate_dense_reference_points,
    shift_reference_points,
)

P2PResidualAdapterBlock = PDCResidualAdapterBlock
P2PFeatureAdapter = DenseCountingFeatureAdapter
RegressionModel = DensePointRegressionHead
ClassificationModel = DensePointClassificationHead
AnchorPoints = DenseReferencePoints
SAM3P2PBranch = PixelLevelDenseCounter
generate_anchor_points = generate_dense_reference_points
shift = shift_reference_points

__all__ = [
    "PDCResidualAdapterBlock",
    "DenseCountingFeatureAdapter",
    "PDCFeatureAdapter",
    "DensePointRegressionHead",
    "PointOffsetHead",
    "DensePointClassificationHead",
    "ForegroundClassificationHead",
    "DenseReferencePoints",
    "PixelLevelDenseCounter",
    "generate_dense_reference_points",
    "shift_reference_points",
    "P2PResidualAdapterBlock",
    "P2PFeatureAdapter",
    "RegressionModel",
    "ClassificationModel",
    "AnchorPoints",
    "SAM3P2PBranch",
    "generate_anchor_points",
    "shift",
]

"""Region-level sparse counter (RSC) losses."""

from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Points
from sam3.train.matcher import DensePointHungarianMatcher

RSCBoxLoss = Boxes
RSCPointLoss = Points
RSCQualityBCELoss = IABCEMdetr
RSCMatcher = DensePointHungarianMatcher
RegionSparseMatcher = DensePointHungarianMatcher

__all__ = [
    "RSCBoxLoss",
    "RSCPointLoss",
    "RSCQualityBCELoss",
    "RSCMatcher",
    "RegionSparseMatcher",
]

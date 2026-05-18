"""Legacy P2P loss import path.

Public code should import `count_anything.train.loss.pdc_loss` directly. This
module only keeps older configs importable.
"""

from count_anything.train.loss.pdc_loss import (  # noqa: F401
    PDCHungarianMatcher,
    PDCPointSetLoss,
    PDCSparseMatcher,
    SparsePointMatcher,
)

P2PSparseMatcher = PDCSparseMatcher
P2POriginalHungarianMatcher = PDCHungarianMatcher
P2PBranchLoss = PDCPointSetLoss

__all__ = [
    "PDCSparseMatcher",
    "PDCHungarianMatcher",
    "PDCPointSetLoss",
    "SparsePointMatcher",
    "P2PSparseMatcher",
    "P2POriginalHungarianMatcher",
    "P2PBranchLoss",
]

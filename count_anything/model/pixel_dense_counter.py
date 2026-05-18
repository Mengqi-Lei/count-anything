"""Pixel-level dense counter (PDC)."""

import numpy as np
import torch
from torch import nn


def _resolve_group_norm_groups(channels, preferred_groups):
    groups = min(preferred_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class PDCResidualAdapterBlock(nn.Module):
    def __init__(self, channels, gn_groups=32):
        super().__init__()
        groups = _resolve_group_norm_groups(channels, gn_groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)
        return self.act(out + residual)


class DenseCountingFeatureAdapter(nn.Module):
    def __init__(
        self,
        channels=256,
        num_blocks=3,
        use_coordconv=False,
        gn_groups=32,
    ):
        super().__init__()
        self.use_coordconv = use_coordconv
        self.num_blocks = num_blocks

        if use_coordconv:
            groups = _resolve_group_norm_groups(channels, gn_groups)
            self.input_proj = nn.Sequential(
                nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
            )
        else:
            self.input_proj = nn.Identity()

        self.blocks = nn.Sequential(
            *[
                PDCResidualAdapterBlock(channels=channels, gn_groups=gn_groups)
                for _ in range(num_blocks)
            ]
        )

    def _make_coord_channels(self, x):
        _, _, height, width = x.shape
        ys = torch.linspace(-1.0, 1.0, steps=height, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, steps=width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        return coords

    def forward(self, x):
        if self.use_coordconv:
            x = torch.cat([x, self._make_coord_channels(x)], dim=1)
        x = self.input_proj(x)
        return self.blocks(x)


class DensePointRegressionHead(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size, num_anchor_points * 2, kernel_size=3, padding=1
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 2)


class DensePointClassificationHead(nn.Module):
    def __init__(
        self,
        num_features_in,
        num_anchor_points=4,
        num_classes=2,
        prior=0.01,
        feature_size=256,
    ):
        super().__init__()
        _ = prior
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.output(out)
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, _ = out1.shape
        out2 = out1.view(
            batch_size, width, height, self.num_anchor_points, self.num_classes
        )
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


def generate_dense_reference_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    reference_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    return reference_points


def shift_reference_points(shape, stride, reference_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    num_reference_points = reference_points.shape[0]
    num_locations = shifts.shape[0]
    all_reference_points = (
        reference_points.reshape((1, num_reference_points, 2))
        + shifts.reshape((1, num_locations, 2)).transpose((1, 0, 2))
    )
    return all_reference_points.reshape((num_locations * num_reference_points, 2))


class DenseReferencePoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super().__init__()
        self.pyramid_levels = [3, 4, 5, 6, 7] if pyramid_levels is None else pyramid_levels
        self.strides = (
            [2**x for x in self.pyramid_levels] if strides is None else strides
        )
        self.row = row
        self.line = line

    def forward(self, image_batch):
        image_shape = np.array(image_batch.shape[2:])
        image_shapes = [(image_shape + stride - 1) // stride for stride in self.strides]
        all_reference_points = np.zeros((0, 2), dtype=np.float32)
        for idx, stride in enumerate(self.strides):
            reference_points = generate_dense_reference_points(
                stride, row=self.row, line=self.line
            )
            shifted_reference_points = shift_reference_points(
                image_shapes[idx], stride, reference_points
            )
            all_reference_points = np.append(
                all_reference_points, shifted_reference_points, axis=0
            )
        all_reference_points = np.expand_dims(all_reference_points, axis=0)
        return torch.from_numpy(all_reference_points).to(
            device=image_batch.device, dtype=image_batch.dtype
        )


class PixelLevelDenseCounter(nn.Module):
    def __init__(
        self,
        row=2,
        line=2,
        stride=7,
        use_feature_adapter=False,
        feature_adapter_num_blocks=3,
        feature_adapter_use_coordconv=False,
        feature_adapter_gn_groups=32,
    ):
        super().__init__()
        self.row = row
        self.line = line
        self.stride = stride
        self.use_feature_adapter = use_feature_adapter
        num_reference_points = row * line
        self.regression = DensePointRegressionHead(
            num_features_in=256, num_anchor_points=num_reference_points
        )
        self.classification = DensePointClassificationHead(
            num_features_in=256,
            num_classes=2,
            num_anchor_points=num_reference_points,
        )
        self.reference_points = DenseReferencePoints(strides=[stride], row=row, line=line)
        self.feature_adapter = (
            DenseCountingFeatureAdapter(
                channels=256,
                num_blocks=feature_adapter_num_blocks,
                use_coordconv=feature_adapter_use_coordconv,
                gn_groups=feature_adapter_gn_groups,
            )
            if use_feature_adapter
            else None
        )

    def _get_pdc_feature(self, pixel_embed, image_batch):
        _ = image_batch
        if self.feature_adapter is not None:
            return self.feature_adapter(pixel_embed)
        return pixel_embed

    def forward(self, pixel_embed, image_batch):
        adapted_feature = self._get_pdc_feature(pixel_embed, image_batch)
        batch_size = adapted_feature.shape[0]
        regression = self.regression(adapted_feature) * 100.0
        classification = self.classification(adapted_feature)
        reference_points = self.reference_points(image_batch).repeat(batch_size, 1, 1)
        output_coord = regression + reference_points
        image_size = image_batch.new_tensor(image_batch.shape[-2:]).view(1, 2).repeat(
            batch_size, 1
        )
        outputs = {
            "pdc_logits": classification,
            "pdc_points": output_coord,
            "pdc_features": adapted_feature,
            "input_image_size": image_size,
        }
        return outputs


PDCFeatureAdapter = DenseCountingFeatureAdapter
PointOffsetHead = DensePointRegressionHead
ForegroundClassificationHead = DensePointClassificationHead


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
]

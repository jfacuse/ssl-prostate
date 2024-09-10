from re import L
import numpy as np
from monai import transforms
from torch import scalar_tensor, zero_
from monai.transforms.transform import LazyTransform
from monai.transforms.inverse import InvertibleTransform
import torch
from monai.utils.enums import TraceKeys
from typing import Sequence
from monai.data.meta_tensor import MetaTensor
from monai.utils import (
    InterpolateMode,
    ensure_tuple,
    ensure_tuple_size,
    fall_back_tuple,
)
from monai.utils.module import look_up_option
from monai.utils.enums import TransformBackends
from monai.config import DtypeLike
from monai.utils.type_conversion import get_equivalent_dtype

import warnings

import numpy as np
import torch

from monai.config import USE_COMPILED
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import AFFINE_TOL, compute_shape_offset, to_affine_nd
from monai.networks.layers import AffineTransform
from monai.transforms.croppad.array import ResizeWithPadOrCrop
from monai.transforms.intensity.array import GaussianSmooth
from monai.transforms.inverse import TraceableTransform
from monai.transforms.utils import create_rotate, create_translate, resolves_modes, scale_affine
from monai.transforms.utils_pytorch_numpy_unification import allclose
from monai.utils import (
    TraceKeys,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)

class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.concatenate(result, axis=0).astype(np.float32)
        return d

def get_scratch_train_transforms(args):
    if args.dataset == 'btcv':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                transforms.RandScaleIntensityd(keys="image",
                                            factors=0.1,
                                            prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image",
                                            offsets=0.1,
                                            prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1, neg=1, num_samples=args.num_samples,
                    image_key="image", image_threshold=0),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                # transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(args.roi_x, args.roi_y, args.roi_z), random_size=False),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"])
                # transforms.EnsureTyped(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'prostate':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                #transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                #transforms.Spacingd(
                #    keys=["image", "label"],
                #    pixdim=(1.0, 1.0, 1.0),
                #    mode=("bilinear", "nearest"),
                #),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                #transforms.RandCropByPosNegLabeld(
                #    keys=["image", "label"], label_key="label",
                #   spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                #    pos=1, neg=1, num_samples=args.num_samples,
                #    image_key="image", image_threshold=0),
                #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                # transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(args.roi_x, args.roi_y, args.roi_z), random_size=False),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"])
                # transforms.EnsureTyped(keys=["image", "label"]),
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return train_transform

def get_mae_pretrain_transforms(args):
    if args.dataset == 'btcv':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset in ['msd_brats']:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1, neg=1, num_samples=args.num_samples,
                    image_key="image", image_threshold=0),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                # transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    elif args.dataset in ['prostate']:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                #transforms.Spacingd(
                #    keys=["image", "label"],
                #    pixdim=(1.0, 1.0, 1.0),
                #    mode=("bilinear", "nearest"),
                #),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                #transforms.RandCropByPosNegLabeld(
                #    keys=["image", "label"], label_key="label",
                #    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                #    pos=1, neg=1, num_samples=args.num_samples,
                #    image_key="image", image_threshold=0),
                # transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return train_transform

def get_val_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset in ['msd_brats']:
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    elif args.dataset in ['prostate']:
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                #transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                #transforms.Spacingd(keys=["image", "label"],
                #                    pixdim=(1.0, 1.0, 1.0),
                #                    mode=("bilinear", "nearest")),
                #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

def get_vis_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.RandCropByPosNegLabeld(
                #     keys=["image", "label"],
                #     label_key="label",
                #     spatial_size=(args.roi_x, args.roi_y, args.roi_x),
                #     pos=1,
                #     neg=1,
                #     num_samples=1,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                transforms.CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z)
                ),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset in ['msd_brats']:
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.RandCropByPosNegLabeld(
                #     keys=["image", "label"],
                #     label_key="label",
                #     spatial_size=(args.roi_x, args.roi_y, args.roi_x),
                #     pos=1,
                #     neg=1,
                #     num_samples=1,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                transforms.CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z)
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    elif args.dataset in ['prostate']:
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                #transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                #transforms.Spacingd(keys=["image", "label"],
                #                    pixdim=(1.0, 1.0, 1.0),
                #                    mode=("bilinear", "nearest")),
                #transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

def get_raw_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset in ['msd_brats', 'prostate']:
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

class Resize():
    def __init__(self, scale_params):
        self.scale_params = scale_params

    def __call__(self, img):
        scale_params = self.scale_params
        shape = img.shape[1:]
        assert len(scale_params) == len(shape)
        spatial_size = []
        for scale, shape_dim in zip(scale_params, shape):
            spatial_size.append(int(scale * shape_dim))
        transform = transforms.Resize(spatial_size=spatial_size, mode='nearest')
        # import pdb
        # pdb.set_trace()
        return transform(img)

def get_post_transforms(args):
    if args.dataset == 'btcv':
        if args.test:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
        else:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
    elif args.dataset in ['msd_brats']:
        post_pred = transforms.Compose([transforms.EnsureType(), transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)])
        post_label = transforms.Identity()
    elif args.dataset in ['prostate']:
        post_pred = transforms.Compose([transforms.EnsureType(), transforms.Activations(sigmoid=True)])
        post_label = transforms.Identity()
    return post_pred, post_label

# Aqui el unico cambio es quitarle el unsqueeze al resize :)

def _maybe_new_metatensor(img, dtype=None, device=None):
    """create a metatensor with fresh metadata if track_meta is True otherwise convert img into a torch tensor"""
    return convert_to_tensor(
        img.as_tensor() if isinstance(img, MetaTensor) else img,
        dtype=dtype,
        device=device,
        track_meta=get_track_meta(),
        wrap_sequence=True,
    )

def resize(
    img, out_size, mode, align_corners, dtype, input_ndim, anti_aliasing, anti_aliasing_sigma, lazy, transform_info
):
    """
    Functional implementation of resize.
    This function operates eagerly or lazily according to
    ``lazy`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        out_size: expected shape of spatial dimensions after resize operation.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
            ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'.
        dtype: data type for resampling computation. If None, use the data type of input data.
        input_ndim: number of spatial dimensions.
        anti_aliasing: whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
        lazy: a flag that indicates whether the operation should be performed lazily or not
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    img = convert_to_tensor(img, track_meta=get_track_meta())
    orig_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    extra_info = {
        "mode": mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "dtype": str(dtype)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
        "new_dim": len(orig_size) - input_ndim,
    }
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=out_size,
        affine=scale_affine(orig_size, out_size),
        extra_info=extra_info,
        orig_size=orig_size,
        transform_info=transform_info,
        lazy=lazy,
    )
    if lazy:
        if anti_aliasing and lazy:
            warnings.warn("anti-aliasing is not compatible with lazy evaluation.")
        out = _maybe_new_metatensor(img)
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    if tuple(convert_to_numpy(orig_size)) == out_size:
        out = _maybe_new_metatensor(img, dtype=torch.float32)
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
    out = _maybe_new_metatensor(img)
    img_ = convert_to_tensor(out, dtype=dtype, track_meta=False)  # convert to a regular tensor
    if anti_aliasing and any(x < y for x, y in zip(out_size, img_.shape[1:])):
        factors = torch.div(torch.Tensor(list(img_.shape[1:])), torch.Tensor(out_size))
        if anti_aliasing_sigma is None:
            # if sigma is not given, use the default sigma in skimage.transform.resize
            anti_aliasing_sigma = torch.maximum(torch.zeros(factors.shape), (factors - 1) / 2).tolist()
        else:
            # if sigma is given, use the given value for downsampling axis
            anti_aliasing_sigma = list(ensure_tuple_rep(anti_aliasing_sigma, len(out_size)))
            for axis in range(len(out_size)):
                anti_aliasing_sigma[axis] = anti_aliasing_sigma[axis] * int(factors[axis] > 1)
        anti_aliasing_filter = GaussianSmooth(sigma=anti_aliasing_sigma)
        img_ = convert_to_tensor(anti_aliasing_filter(img_), track_meta=False)
    _, _m, _, _ = resolves_modes(mode, torch_interpolate_spatial_nd=len(img_.shape) - 1)
    resized = torch.nn.functional.interpolate(
        input=img_, size=out_size, mode=mode, align_corners=align_corners
    )
    out, *_ = convert_to_dst_type(resized, out, dtype=torch.float32)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out

class Resize(InvertibleTransform, LazyTransform):
    """
    Resize the input image to given spatial size (with scaling, not cropping/padding).
    Implemented using :py:class:`torch.nn.functional.interpolate`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        spatial_size: expected shape of spatial dimensions after resize operation.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        size_mode: should be "all" or "longest", if "all", will use `spatial_size` for all the spatial dims,
            if "longest", rescale the image so that only the longest side is equal to specified `spatial_size`,
            which must be an int number in this case, keeping the aspect ratio of the initial image, refer to:
            https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
            #albumentations.augmentations.geometric.resize.LongestMaxSize.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        anti_aliasing: bool
            Whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
            By default, this value is chosen as (s - 1) / 2 where s is the
            downsampling factor, where s > 1. For the up-size case, s < 1, no
            anti-aliasing is performed prior to rescaling.
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        spatial_size: Sequence[int] | int,
        size_mode: str = "all",
        mode: str = InterpolateMode.AREA,
        align_corners: bool | None = None,
        anti_aliasing: bool = False,
        anti_aliasing_sigma: Sequence[float] | float | None = None,
        dtype: DtypeLike | torch.dtype = torch.float32,
        lazy: bool = False,
    ) -> None:
        LazyTransform.__init__(self, lazy=lazy)
        self.size_mode = look_up_option(size_mode, ["all", "longest"])
        self.spatial_size = spatial_size
        self.mode = mode
        self.align_corners = align_corners
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = anti_aliasing_sigma
        self.dtype = dtype

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        align_corners: bool | None = None,
        anti_aliasing: bool | None = None,
        anti_aliasing_sigma: Sequence[float] | float | None = None,
        dtype: DtypeLike | torch.dtype = None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
                ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            anti_aliasing: bool, optional
                Whether to apply a Gaussian filter to smooth the image prior
                to downsampling. It is crucial to filter when downsampling
                the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
            anti_aliasing_sigma: {float, tuple of floats}, optional
                Standard deviation for Gaussian filtering used when anti-aliasing.
                By default, this value is chosen as (s - 1) / 2 where s is the
                downsampling factor, where s > 1. For the up-size case, s < 1, no
                anti-aliasing is performed prior to rescaling.
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        Raises:
            ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

        """
        anti_aliasing = self.anti_aliasing if anti_aliasing is None else anti_aliasing
        anti_aliasing_sigma = self.anti_aliasing_sigma if anti_aliasing_sigma is None else anti_aliasing_sigma

        input_ndim = img.ndim - 2  # spatial ndim
        if self.size_mode == "all":
            output_ndim = len(ensure_tuple(self.spatial_size))
            if output_ndim > input_ndim:
                input_shape = ensure_tuple_size(img.shape, output_ndim + 1, 1)
                img = img.reshape(input_shape)
            elif output_ndim < input_ndim:
                raise ValueError(
                    "len(spatial_size) must be greater or equal to img spatial dimensions, "
                    f"got spatial_size={output_ndim} img={input_ndim}."
                )
            _sp = img.shape[2:]
            sp_size = fall_back_tuple(self.spatial_size, _sp)
        else:  # for the "longest" mode
            img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
            if not isinstance(self.spatial_size, int):
                raise ValueError("spatial_size must be an int number if size_mode is 'longest'.")
            scale = self.spatial_size / max(img_size)
            sp_size = tuple(int(round(s * scale)) for s in img_size)

        _mode = self.mode if mode is None else mode
        _align_corners = self.align_corners if align_corners is None else align_corners
        _dtype = get_equivalent_dtype(dtype or self.dtype or img.dtype, torch.Tensor)
        lazy_ = self.lazy if lazy is None else lazy
        return resize(  # type: ignore
            img,
            sp_size,
            _mode,
            _align_corners,
            _dtype,
            input_ndim,
            anti_aliasing,
            anti_aliasing_sigma,
            lazy_,
            self.get_transform_info(),
        )

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)

    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        orig_size = transform[TraceKeys.ORIG_SIZE]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        dtype = transform[TraceKeys.EXTRA_INFO]["dtype"]
        xform = Resize(
            spatial_size=orig_size,
            mode=mode,
            align_corners=None if align_corners == TraceKeys.NONE else align_corners,
            dtype=dtype,
        )
        with xform.trace_transform(False):
            data = xform(data)
        for _ in range(transform[TraceKeys.EXTRA_INFO]["new_dim"]):
            data = data.squeeze(-1)  # remove the additional dims
        return data
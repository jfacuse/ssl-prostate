import numpy as np

import torch
import torchvision
from itertools import repeat
import collections.abc
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

import pdb

def patches3d_to_grid(patches, patch_size=16, grid_size=8, in_chans=4, n_group=3, hidden_axis='d', slice_pos_list=[0.4, 0.45, 0.5, 0.55, 0.6]):
    """
    input patches are in 3D which contain height, width and depth
    -------
    Params:
    --patches: [B, L, C*H*W*D]
    --patch_size: 
    --grid_size:
    --in_chans:
    --n_groups: group number of patches, e.g., original patch group, masked patch group, recon patch group
    --hidden_axis: indicate the axis to be hidden because we can only visualize a 2D image instead of 3D volume
    """
    B, L, C = patches.shape
    print('Patches Shape:', patches.shape,B,L,C)
    patch_size = to_3tuple(patch_size)
    if isinstance(grid_size, int):
        grid_size = to_3tuple(grid_size)
    assert np.prod(grid_size) == L and np.prod(patch_size) * in_chans == C, "Shape of input doesn't match parameters"
    print(f"grid_size: {grid_size}, patch_size: {patch_size}")

    patches = patches.reshape(B, *grid_size, *patch_size, in_chans)
    # restore image structure
    image = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, in_chans, 1,
                                                            grid_size[0] * patch_size[0], 
                                                            grid_size[1] * patch_size[1], 
                                                            grid_size[2] * patch_size[2])

    assert B % n_group == 0
    n_per_row = len(slice_pos_list) * in_chans * B // n_group
    # always choose the specified slice to visualize
    if hidden_axis == 'd':
        slice_list = []
        for slice_pos in slice_pos_list:
            slice_list.append(image[..., :, :, int(image.size(-1) * slice_pos)])
            print('Print slice', int(image.size(-1) * slice_pos))
        image = torch.cat(slice_list, dim=2)
    elif hidden_axis == 'h':
        slice_list = []
        for slice_pos in slice_pos_list:
            slice_list.append(image[..., int(image.size(-3) * slice_pos), :, :])
            print('Print slice', int(image.size(-3) * slice_pos))
        image = torch.cat(slice_list, dim=2)
    else:
        raise ValueError(f"Only support D for now")
    visH, visW = image.size(-2), image.size(-1)
    print(visH, visW, image.shape)
    grid_of_images = torchvision.utils.make_grid(image.reshape(B * len(slice_pos_list) * in_chans, 1, visH, visW), nrow=n_per_row)
    #grid_of_images.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
    #v_min = grid_of_images.min()
    #v_max = grid_of_images.max()
    #nmin, nmax = 0, 255
    #grid_of_images = (grid_of_images - v_min)/(v_max - v_min)*(nmax - nmin) + nmin
    grid_of_images.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #grid_of_images = np.fix(grid_of_images)
    #print(grid_of_images.max())

    #grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    return grid_of_images

def images3d_to_grid(image, n_group=3, hidden_axis='d', slice_pos_list=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    input patches are in 3D which contain height, width and depth
    -------
    Params:
    --image: [B, C, H, W, D]
    --n_groups: group number of patches, e.g., original patch group, masked patch group, recon patch group
    --hidden_axis: indicate the axis to be hidden because we can only visualize a 2D image instead of 3D volume
    """
    B, C, H, W, D = image.shape

    assert B % n_group == 0
    n_per_row = B // n_group
    list_of_grid_images = []
    for slice_pos in slice_pos_list:
        if hidden_axis == 'd':
            image_slice = image[..., :, :, int(D * slice_pos)] # [B, 3, H, W]
        else:
            raise ValueError(f"Only support D for now")
        # pdb.set_trace()
        grid_of_images = torchvision.utils.make_grid(image_slice, nrow=n_per_row)
        grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        list_of_grid_images.append(grid_of_images)

    return list_of_grid_images

def images3d_to_grid_alt(image, in_chans=3, n_group=3, hidden_axis='h', slice_pos_list=[0.4, 0.45, 0.5, 0.55, 0.6]):
    """
    input patches are in 3D which contain height, width and depth
    -------
    Params:
    --patches: [B, L, C*H*W*D]
    --patch_size: 
    --grid_size:
    --in_chans:
    --n_groups: group number of patches, e.g., original patch group, masked patch group, recon patch group
    --hidden_axis: indicate the axis to be hidden because we can only visualize a 2D image instead of 3D volume
    """
    print(image.shape)
    B, C, H, W, D = image.shape
    assert B % n_group == 0
    n_per_row = len(slice_pos_list) * in_chans * B // n_group
    # always choose the specified slice to visualize
    image = image.reshape(B,in_chans,1,H,W,D)
    if hidden_axis == 'd':
        slice_list = []
        for slice_pos in slice_pos_list:
            slice_list.append(image[..., :, :, int(image.size(-1) * slice_pos)])
            print('Print slice', int(image.size(-1) * slice_pos))
        image = torch.cat(slice_list, dim=2)
    elif hidden_axis == 'h':
        slice_list = []
        for slice_pos in slice_pos_list:
            slice_list.append(image[..., int(image.size(-3) * slice_pos), :, :])
            print('Print slice', int(image.size(-3) * slice_pos))
        image = torch.cat(slice_list, dim=2)
    else:
        raise ValueError(f"Only support D for now")
    visH, visW = image.size(-2), image.size(-1)
    print(visH, visW, image.shape)
    grid_of_images = torchvision.utils.make_grid(image.reshape(B * len(slice_pos_list) * in_chans, 1, visH, visW), nrow=n_per_row)
    #grid_of_images.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
    #v_min = grid_of_images.min()
    #v_max = grid_of_images.max()
    #nmin, nmax = 0, 255
    #grid_of_images = (grid_of_images - v_min)/(v_max - v_min)*(nmax - nmin) + nmin
    grid_of_images.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #grid_of_images = np.fix(grid_of_images)
    #print(grid_of_images.max())

    #grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    return grid_of_images
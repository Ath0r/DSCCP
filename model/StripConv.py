# -*- coding: utf-8 -*-
import os
from re import X
import torch
from torch import nn
import einops


"""Dynamic Strip Convolution Module"""

class StripConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        extend_scope,
        if_offset,
        device
    ):
        """
        A Dynamic Strip Convolution Implementation

        Based on:

            Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation
            https://github.com/YaoleiQi/DSCNet

        Args:
            in_ch: number of input channels. Defaults to 1.
            out_ch: number of output channels. Defaults to 1.
            kernel_size: the size of kernel. Defaults to 9.
            extend_scope: the range to expand. Defaults to 1 for this method.
            if_offset: whether deformation is required,  if it is False, it is the standard convolution kernel. Defaults to True.

        """

        super().__init__()

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.gn_offset = nn.GroupNorm(3, kernel_size)

        self.gn_theta = nn.GroupNorm(3, kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        # Convolution for learning the vertical component offset
        self.offset_conv = nn.Conv2d(in_channels, kernel_size, 3, padding=1)

        # Convolution for learning the angle θ
        self.theta_conv = nn.Conv2d(in_channels, kernel_size, 5, padding=2) 

        self.strip_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # Predict offset map between [-1, 1]
        offset = self.offset_conv(input)
        # Initialize offset
        # size = (input.shape[0], self.kernel_size, input.shape[2], input.shape[3])
        # offset = torch.rand(size)
        # offset = offset * 2 - 1 
        # offset = offset.to(self.device)   # Move tensor to GPU

        # offset = self.bn(offset)
        offset = self.gn_offset(offset) 
        offset = self.tanh(offset)

        # Predict theta map between [-π/2， π/2]
        offset_theta = self.theta_conv(input)
        offset_theta = self.gn_theta(offset_theta)
        offset_theta = torch.atan(offset_theta)

        # Run deformative conv
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            offset_theta=offset_theta,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        output = self.strip_conv(deformed_feature)


        # Groupnorm & ReLU
        output = self.gn(output)
        output = self.relu(output)

        return output


def get_coordinate_map_2D(
    offset,
    offset_theta,
    extend_scope,
    device
):
    """Computing 2D coordinate map of DSCNet based on: TODO

    Args:
        offset: offset created by random 
        offset_theta: theta offset predict by network with shape [B,2*K, W, H] .Here K refers to kernel size.
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """


    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1]
    center = kernel_size // 2
    device = torch.device(device)


    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    
    """
    Initialize the kernel and flatten the kernel
    y: only need 0
    x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
    """
    y_spread_ = torch.linspace(-center, center, kernel_size, device=device)    
    x_spread_ = torch.linspace(-center, center, kernel_size, device=device)  

    y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
    x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

    y_new_ = y_center_ + y_grid_
    x_new_ = x_center_ + x_grid_

    y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
    x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

    offset = einops.rearrange(offset, "b k w h -> k b w h")

    offset_theta = einops.rearrange(offset_theta, "b k w h -> k b w h")

    y_offset_new_ = offset.detach().clone()
    
    x_offset_new_ = offset.detach().clone()

    # The center position remains unchanged and the rest of the positions begin to swing
    # This part is quite simple. The main idea is that "offset is an iterative process"

    y_offset_new_[center] = 0
    x_offset_new_[center] = 0

    for index in range(1, center + 1):
        y_offset_new_[center + index] = (
            index * torch.tan(offset_theta[center + index]) + offset[center + index] * torch.cos(offset_theta[center + index])
        )
        y_offset_new_[center - index] = (
            - index * torch.tan(offset_theta[center + index]) + offset[center + index] * torch.cos(offset_theta[center + index])
        )

    for index in range(1, center + 1):
        x_offset_new_[center + index] = (
            index - torch.sin(offset_theta[center + index]) * offset[center + index]
        )
        x_offset_new_[center - index] = (
            -index - torch.sin(offset_theta[center + index]) * offset[center + index]
        )
 
    y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")
    x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

    y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))
    x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

    y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
    x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

   
    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.

    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """Map the value of coordinate_map from origin=[min, max] to target=[a,b] for DSCNet based on: TODO

    Args:
        coordinate_map: the coordinate map to be scaled
        origin: original value range of coordinate map, e.g. [coordinate_map.min(), coordinate_map.max()]
        target: target value range of coordinate map,Defaults to [-1, 1]

    Return:
        coordinate_map_scaled: the coordinate map after scaling
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled

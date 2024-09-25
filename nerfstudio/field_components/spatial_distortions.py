# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Space distortions."""

import abc
from typing import Optional, Union

import torch
from functorch import jacrev, vmap
from jaxtyping import Float
from torch import Tensor, nn
import einops

from nerfstudio.utils.math import Gaussians


class SpatialDistortion(nn.Module):
    """Apply spatial distortions"""

    @abc.abstractmethod
    def forward(self, positions: Union[Float[Tensor, "*bs 3"], Gaussians]) -> Union[Float[Tensor, "*bs 3"], Gaussians]:
        """
        Args:
            positions: Sample to distort

        Returns:
            Union: distorted sample
        """


class SceneContraction(SpatialDistortion):
    """Contract unbounded space using the contraction was proposed in MipNeRF-360.
        We use the following contraction equation:

        .. math::

            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}

        If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
        radius 2. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 4.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.

        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.

    """

    def __init__(self, order: Optional[Union[float, int]] = None) -> None:
        super().__init__()
        self.order = order

    @staticmethod
    def static_forward(order, positions):
        def contract(x):
            mag = torch.linalg.norm(x, ord=order, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        if isinstance(positions, Gaussians):
            means = contract(positions.mean.clone())

            def contract_gauss(x):
                return (2 - 1 / torch.linalg.norm(x, ord=order, dim=-1, keepdim=True)) * (
                    x / torch.linalg.norm(x, ord=order, dim=-1, keepdim=True)
                )

            jc_means = vmap(jacrev(contract_gauss))(positions.mean.view(-1, positions.mean.shape[-1]))
            jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])

            # Only update covariances on positions outside the unit sphere
            mag = positions.mean.norm(dim=-1)
            mask = mag >= 1
            cov = positions.cov.clone()
            cov[mask] = jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)

            return Gaussians(mean=means, cov=cov)

        return contract(positions)

    @staticmethod
    def static_undo_forward(order, positions):
        def undo_contract(y):
            y_mag = torch.linalg.norm(y, ord=order, dim=-1)[..., None]
            return torch.where(
                y_mag < 1, 
                y, 
                (1 / (2 - y_mag)) * (y / y_mag),
            )
        
        if isinstance(positions, Gaussians):
            raise NotImplementedError("Undo contraction not implemented for Gaussians")
        
        return undo_contract(positions)
    
    @staticmethod
    def static_undo_forward_1d(order, positions):
        # same as static_undo_forward but for 1d positions
        # (the position does not have x, y, z) it only has a floating point, the rest is assumed 0
        def undo_contract_1d(y):
            y_mag = torch.abs(y)
            return torch.where(
                y_mag < 1, 
                y, 
                (1 / (2 - y_mag)) * torch.sign(y),
            )
        if isinstance(positions, Gaussians):
            raise NotImplementedError("Undo contraction not implemented for Gaussians")
        return undo_contract_1d(positions)

    def forward(self, positions):
        return SceneContraction.static_forward(self.order, positions)

    def undo_forward(self, positions):
        return SceneContraction.static_undo_forward(self.order, positions)
    
    def undo_forward_1d(self, positions):
        return SceneContraction.static_undo_forward_1d(self.order, positions)


def visualize_occupancy_grid(occupancy_grid):
    for z in range(occupancy_grid.shape[2]):
        # Check if the layer contains any True values
        if not torch.any(occupancy_grid[:, :, z]):
            print(f"Layer {z+1} is all o\n")
            continue
        
        print(f"Layer {z+1}:")
        for y in range(occupancy_grid.shape[1]):
            line = ""
            for x in range(occupancy_grid.shape[0]):
                line += "x" if occupancy_grid[x, y, z] else "o"
            print(line)
        print("\n")  # Extra newline for spacing between layers


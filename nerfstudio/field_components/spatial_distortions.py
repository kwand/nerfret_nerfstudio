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

    def forward(self, positions):
        return SceneContraction.static_forward(self.order, positions)

    def undo_forward(self, positions):
        return SceneContraction.static_undo_forward(self.order, positions)


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


def main():
    # this is the number of voxels in the grid (not the number of lines)
    grid_voxel_num = 11
    grid_line_num = grid_voxel_num + 1
    
    # the occupancy grid to be filled
    occupancy_grid = torch.zeros((grid_voxel_num, grid_voxel_num, grid_voxel_num), dtype=torch.bool)
    
    # linspace from -2 to 2 with 64 points
    grid_delta = 4 / grid_voxel_num
    grid_locations_warpped_1d = torch.linspace(-2, 2, grid_line_num)
    
    # the -2 and 2 at the end correspond to infinity in the uncontracted space
    # causing computation issues, so we use the middle of the last voxel as a proxy
    grid_locations_warpped_1d[0] = grid_locations_warpped_1d[0] + grid_delta / 2
    grid_locations_warpped_1d[-1] = grid_locations_warpped_1d[-1] - grid_delta / 2
    
    grid_locations_warpped = torch.zeros((grid_line_num, 3))
    grid_locations_warpped[:, 0] = grid_locations_warpped_1d
    
    grid_locations_uncontracted = SceneContraction(order=torch.inf).undo_forward(grid_locations_warpped)
    grid_locations_uncontracted_1d = grid_locations_uncontracted[:, 0]
    
    origin = torch.tensor([[-0.1, -0.2, -0.3], [-0.12, -0.22, -0.32]])
    direction = torch.tensor([[0.33, 0.0, 0.12], [0.2, -0.4, 0.1]])
    
    _, max_direction_indices = torch.max(torch.abs(direction), dim=1)
    batch_indices = torch.arange(0, origin.shape[0])
    
    # fastest moving direction is x, calculate where x hits the grid
    all_t = (grid_locations_uncontracted_1d.unsqueeze(1) - \
            origin[batch_indices, max_direction_indices].unsqueeze(0)) \
            / direction[batch_indices, max_direction_indices].unsqueeze(0)
    
    all_x = origin[:, 0] + all_t * direction[:, 0]
    all_y = origin[:, 1] + all_t * direction[:, 1]
    all_z = origin[:, 2] + all_t * direction[:, 2]
    
    all_t = all_t.T
    all_x = all_x.T
    all_y = all_y.T
    all_z = all_z.T
    
    ray_locations_uncontracted = torch.stack([all_x, all_y, all_z], dim=-1)
    ray_locations_warpped = SceneContraction(order=torch.inf).forward(ray_locations_uncontracted)
    
    grid_location_neg_to_pos = ray_locations_warpped / grid_delta
    grid_location_zero_to_pos = grid_location_neg_to_pos + grid_voxel_num / 2
    
    # these are the grid voxels (same or above the max direction intersection line)
    grid_location_zero_to_pos_int = torch.floor(grid_location_zero_to_pos).int()
    
    # these are the grid voxels (same or below the max direction intersection line)
    # end is excluded because the intersection voxel at infinity should not be included
    grid_location_zero_to_pos_int_end_excluded = grid_location_zero_to_pos_int[:, 1:-1, :]
    moved_by_1_in_max_direction = grid_location_zero_to_pos_int_end_excluded.clone()
    moved_by_1_in_max_direction[batch_indices[:, None], :, max_direction_indices[:, None]] -= 1
    
    assert torch.all(grid_location_zero_to_pos_int >= 0), f"{grid_location_zero_to_pos_int.min()}"
    assert torch.all(moved_by_1_in_max_direction >= 0), f"{moved_by_1_in_max_direction.min()}"
    
    # fill in grid voxels above the intersection line
    mask_above = (all_t > 0).unsqueeze(-1).expand_as(grid_location_zero_to_pos_int)
    occupied_above = grid_location_zero_to_pos_int[mask_above].view(-1, 3).t()
    occupancy_grid.index_put_(tuple(occupied_above), torch.tensor(1, dtype=torch.bool), accumulate=False)
    
    # fill in grid voxels below the intersection line
    dummy = 1
    mask_below = mask_above[:, 1:-1, :]
    occupied_below = moved_by_1_in_max_direction[mask_below].view(-1, 3).t()
    occupancy_grid.index_put_(tuple(occupied_below), torch.tensor(1, dtype=torch.bool), accumulate=False)
    
    # warp the x, y and z back
    # ray_locations_uncontracted = torch.stack([x, y, z], dim=1)
    # ray_locations_warpped = SceneContraction(order=torch.inf).forward(ray_locations_uncontracted)
    
    print(f"{occupancy_grid.sum()=}")
    visualize_occupancy_grid(occupancy_grid)
    dummy = 1


if __name__ == "__main__":
    main()
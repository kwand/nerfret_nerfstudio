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

def ray_voxel_intersection(origins, directions, depths, voxel_size, voxel_grid):
    grid_delta = voxel_size
    grid_voxel_num = voxel_grid.shape[0]
    grid_line_num = grid_voxel_num + 1
    grid_locations_warpped_1d = torch.linspace(-2, 2, grid_line_num)

    # the -2 and 2 at the end correspond to infinity in the uncontracted space
    # causing computation issues, so we use the middle of the last voxel as a proxy
    grid_locations_warpped_1d[0] = grid_locations_warpped_1d[0] + grid_delta / 2
    grid_locations_warpped_1d[-1] = grid_locations_warpped_1d[-1] - grid_delta / 2

    grid_locations_warpped = torch.zeros((grid_line_num, 3))
    grid_locations_warpped[:, 0] = grid_locations_warpped_1d

    grid_locations_uncontracted = SceneContraction(order=torch.inf).undo_forward(grid_locations_warpped)
    grid_locations_uncontracted_1d = grid_locations_uncontracted[:, 0]

    _, max_direction_indices = torch.max(torch.abs(directions), dim=1)
    batch_indices = torch.arange(0, origins.shape[0])

    # fastest moving direction is x, calculate where x hits the grid
    all_t = (grid_locations_uncontracted_1d.unsqueeze(1) - \
            origins[batch_indices, max_direction_indices].unsqueeze(0)) \
            / directions[batch_indices, max_direction_indices].unsqueeze(0)
    
    all_x = origins[:, 0] + all_t * directions[:, 0]
    all_y = origins[:, 1] + all_t * directions[:, 1]
    all_z = origins[:, 2] + all_t * directions[:, 2]
    
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

    # these are the grid voxels (same or below the max direction intersection line)
    # end is excluded because the intersection voxel at infinity should not be included
    grid_location_zero_to_pos_int_end_excluded = grid_location_zero_to_pos_int[:, 1:-1, :]
    moved_by_1_in_max_direction = grid_location_zero_to_pos_int_end_excluded.clone()
    moved_by_1_in_max_direction[batch_indices[:, None], :, max_direction_indices[:, None]] -= 1
    
    assert torch.all(grid_location_zero_to_pos_int >= 0), f"{grid_location_zero_to_pos_int.min()}"
    assert torch.all(moved_by_1_in_max_direction >= 0), f"{moved_by_1_in_max_direction.min()}"
    
    # fill in grid voxels above the intersection line
    mask_above = (all_t > 0 & all_t < depths).unsqueeze(-1).expand_as(grid_location_zero_to_pos_int)
    occupied_above = grid_location_zero_to_pos_int[mask_above].view(-1, 3).t()
    voxel_grid.index_put_(tuple(occupied_above), torch.tensor(1, dtype=torch.bool), accumulate=False)

    # fill in grid voxels below the intersection line
    mask_below = mask_above[:, 1:-1, :]
    occupied_below = moved_by_1_in_max_direction[mask_below].view(-1, 3).t()
    voxel_grid.index_put_(tuple(occupied_below), torch.tensor(1, dtype=torch.bool), accumulate=False)

    return voxel_grid

# old implementation, archiving
class DirectionRayVoxelIntersection():
    def __init__(
        self,
        num_voxel: int = 64,
        side_length: float = 4.0,
        device: str = "cpu",
    ):
        self.num_voxel = num_voxel
        self.side_length = side_length
        self.voxel_size = side_length / num_voxel
        self.device = device
        
        self.coverage_map = torch.zeros((num_voxel, num_voxel, num_voxel, 6), dtype=torch.bool, device=self.device)
        
        self.voxel_face_vectors = torch.tensor([
            [1, 0, 0],  # right
            [-1, 0, 0],  # left
            [0, 1, 0],  # top
            [0, -1, 0],  # bottom
            [0, 0, 1],  # front
            [0, 0, -1],  # back
        ], dtype=torch.float32, device=self.device)

    def update_coverage_map(
        self,
        origins,
        directions,
        depths,
    ):
        
        origins = origins.to(self.device)
        directions = directions.to(self.device)
        depths = depths.to(self.device)
        
        grid_delta = self.voxel_size
        grid_voxel_num = self.coverage_map.shape[0]
        grid_line_num = grid_voxel_num + 1
        grid_locations_warpped_1d = torch.linspace(-2, 2, grid_line_num).to(self.device)

        grid_locations_warpped_1d[0] = grid_locations_warpped_1d[0] + grid_delta / 2
        grid_locations_warpped_1d[-1] = grid_locations_warpped_1d[-1] - grid_delta / 2

        grid_locations_warpped = torch.zeros((grid_line_num, 3)).to(self.device)
        grid_locations_warpped[:, 0] = grid_locations_warpped_1d

        grid_locations_uncontracted = SceneContraction(order=torch.inf).undo_forward(grid_locations_warpped)
        grid_locations_uncontracted_1d = grid_locations_uncontracted[:, 0]

        _, max_direction_indices = torch.max(torch.abs(directions), dim=1)
        batch_indices = torch.arange(0, origins.shape[0]).to(self.device)

        all_t = (grid_locations_uncontracted_1d.unsqueeze(1) - \
                origins[batch_indices, max_direction_indices].unsqueeze(0)) \
                / directions[batch_indices, max_direction_indices].unsqueeze(0)
        
        all_x = origins[:, 0] + all_t * directions[:, 0]
        all_y = origins[:, 1] + all_t * directions[:, 1]
        all_z = origins[:, 2] + all_t * directions[:, 2]
        
        all_t = all_t.T
        all_x = all_x.T
        all_y = all_y.T
        all_z = all_z.T
        
        ray_locations_uncontracted = torch.stack([all_x, all_y, all_z], dim=-1)
        ray_locations_warpped = SceneContraction(order=torch.inf).forward(ray_locations_uncontracted)
        
        grid_location_neg_to_pos = ray_locations_warpped / grid_delta
        grid_location_zero_to_pos = grid_location_neg_to_pos + grid_voxel_num / 2
        
        grid_location_zero_to_pos_int = torch.floor(grid_location_zero_to_pos).int()
        
        grid_location_zero_to_pos_int_end_excluded = grid_location_zero_to_pos_int[:, 1:-1, :]
        moved_by_1_in_max_direction = grid_location_zero_to_pos_int_end_excluded.clone()
        moved_by_1_in_max_direction[batch_indices[:, None], :, max_direction_indices[:, None]] -= 1
        dirxfaces = torch.matmul(directions, self.voxel_face_vectors.T) > 0
        assert torch.all(grid_location_zero_to_pos_int >= 0), f"{grid_location_zero_to_pos_int.min()}"
        assert torch.all(moved_by_1_in_max_direction >= 0), f"{moved_by_1_in_max_direction.min()}"
        
        num_rays = origins.shape[0]
        grid_location_zero_to_pos_int_repeated = grid_location_zero_to_pos_int.repeat_interleave(6, dim=1)
        faces = torch.arange(6).repeat(grid_location_zero_to_pos_int.shape[1]).unsqueeze(0).unsqueeze(2).repeat(num_rays, 1, 1).to(self.device)
        grid_location_zero_to_pos_int_faces = torch.cat([grid_location_zero_to_pos_int_repeated, faces], dim=2)

        moved_by_1_in_max_direction_repeated = moved_by_1_in_max_direction.repeat_interleave(6, dim=1)
        faces = torch.arange(6).repeat(moved_by_1_in_max_direction.shape[1]).unsqueeze(0).unsqueeze(2).repeat(num_rays, 1, 1).to(self.device)
        moved_by_1_in_max_direction_faces = torch.cat([moved_by_1_in_max_direction_repeated, faces], dim=2)

        all_t_repeated = all_t.repeat_interleave(6, dim=1)
        dirxfaces_repeated = dirxfaces.repeat(1, grid_location_zero_to_pos_int.shape[1])

        depths_repeated = depths.unsqueeze(1).repeat(1, grid_location_zero_to_pos_int.shape[1]).repeat(1, 6)

        mask_above = ((all_t_repeated > 0) & (all_t_repeated < depths_repeated) & dirxfaces_repeated).unsqueeze(-1).expand_as(grid_location_zero_to_pos_int_faces)
        occupied_above = grid_location_zero_to_pos_int_faces[mask_above].view(-1, 4)
        
        occupied_above = occupied_above.T
        self.coverage_map.index_put_(tuple(occupied_above), torch.tensor(1, dtype=torch.bool), accumulate=False)
        
        mask_below = mask_above[:, 6:-6, :]
        occupied_below = moved_by_1_in_max_direction_faces[mask_below].view(-1, 4).t()
        self.coverage_map.index_put_(tuple(occupied_below), torch.tensor(1, dtype=torch.bool), accumulate=False)
        return self.coverage_map

class DirectionRayVoxelIntersection():
    def __init__(
        self,
        num_voxel: int = 64,
        side_length: float = 4.0,
        device: str = "cpu",
    ):
        self.num_voxel = num_voxel
        self.side_length = side_length
        self.voxel_size = side_length / num_voxel
        self.half_voxel_size = 0.5 * self.voxel_size
        self.device = device
        
        self.coverage_map = torch.zeros((num_voxel, num_voxel, num_voxel, 6), dtype=torch.bool, device=self.device)
        self.do_direction = True
        
        self.voxel_face_vectors = torch.tensor([
            [1, 0, 0],  # right
            [-1, 0, 0],  # left
            [0, 1, 0],  # top
            [0, -1, 0],  # bottom
            [0, 0, 1],  # front
            [0, 0, -1],  # back
        ], dtype=torch.float32, device=self.device).T

    def update_coverage_map(
        self,
        origins,
        directions,
        depths,
    ):
        
        num_rays = origins.shape[0]
        # origins = origins.to(self.device)
        # directions = directions.to(self.device)
        # depths = depths.to(self.device)
        
        # some definition for this function:
        # wrapped: the space where the contraction is applied (the 4 x 4 x 4 box)
        #     rays travel in a curved line in the wrapped space
        # uncontracted: the space where the contraction is not applied (the inf x inf x inf world)
        #     rays travel in a straight line in the uncontracted space
        
        # the number of lines of the grid is 1 more than the number of voxels
        grid_line_num = self.num_voxel + 1

        # vectorized calculation to figure out where the grid lines are
        grid_locations_warpped_1d = torch.linspace(-2, 2, grid_line_num, device=self.device)
        # the -2 and 2 at the end correspond to infinity in the uncontracted space
        # causing computation issues, so we use the middle of the first and last voxel as a proxy
        grid_locations_warpped_1d[0] = grid_locations_warpped_1d[0] + self.half_voxel_size
        grid_locations_warpped_1d[-1] = grid_locations_warpped_1d[-1] - self.half_voxel_size
        # where the grid lines are in the uncontracted space
        grid_locations_uncontracted_1d = SceneContraction(order=torch.inf).undo_forward_1d(
            positions=grid_locations_warpped_1d,
        )

        _, max_direction_indices = torch.max(torch.abs(directions), dim=1)
        max_direction_indices = max_direction_indices.int()
        batch_indices = torch.arange(0, num_rays, device=self.device)

        # all_t.shape is [grid_line_num, batch_size]
        # it signals at what point in time (can be positive or negative),
        # does the ray (# batch_size of them) hit the grid lines (# grid_line_num of them)
        all_t = (grid_locations_uncontracted_1d.unsqueeze(1) - \
                origins[batch_indices, max_direction_indices].unsqueeze(0)) \
                / directions[batch_indices, max_direction_indices].unsqueeze(0)
        all_t = all_t.T
        
        # calculate where does the ray hits the grid in the uncontracted space (calculate all x, y, z)
        # g stands for grid lines, r stands for rays
        ray_locations_uncontracted = origins.unsqueeze(-2) + all_t.unsqueeze(-1) * directions.unsqueeze(-2)
        # the above calculation is equivalent to the following
        # all_x = origins[:, 0] + all_t * directions[:, 0]
        # all_y = origins[:, 1] + all_t * directions[:, 1]
        # all_z = origins[:, 2] + all_t * directions[:, 2]
        # all_t = all_t.T
        # all_x = all_x.T
        # all_y = all_y.T
        # all_z = all_z.T
        # ray_locations_uncontracted = torch.stack([all_x, all_y, all_z], dim=-1)
        
        # warp the ray locations to the wrapped space
        ray_locations_warpped = SceneContraction(order=torch.inf).forward(ray_locations_uncontracted)
        
        # neg: negative index, pos: positive index
        # this is roughly from -32 to 32 for a grid of 64 voxels
        gridloc_neg_to_pos = ray_locations_warpped / self.voxel_size
        # this is roughly from 0 to 64 for a grid of 64 voxels
        gridloc_pos = gridloc_neg_to_pos + self.num_voxel / 2
        # get the integer index, positive
        gridloc_pos_int = torch.floor(gridloc_pos).int()
        assert torch.all(gridloc_pos_int >= 0), f"{gridloc_pos_int.min()}"
        
        # num_rays x 6, bool
        # for each direction, calculate if the ray hits the (six) faces of the voxel
        dirxfaces = torch.matmul(directions, self.voxel_face_vectors) > 0
        dirxfaces_unsquee = einops.rearrange(dirxfaces, 'num_rays six -> num_rays 1 six')
        
        # prepare the grid hit or no hit to create the mask
        all_t_unsquee = einops.rearrange(all_t, 'num_rays grid_lines -> num_rays grid_lines 1')
        depths_unsquee = einops.rearrange(depths, 'num_rays -> num_rays 1 1')
        
        # combine grid hit or no hit with the direction hit or no hit
        grid_and_face_hit_mask = ((all_t_unsquee > 0) & (all_t_unsquee < depths_unsquee) & dirxfaces_unsquee)
        
        # flatten the ray and grid line to prepare for coverage_map update
        gridloc_indices = einops.rearrange(gridloc_pos_int, "num_rays grid_lines three -> (num_rays grid_lines) three")
        grid_and_face_hit_mask_flat = einops.rearrange(grid_and_face_hit_mask, "num_rays grid_lines six -> (num_rays grid_lines) six")
        
        # the grid locations are used for indexing the coverage map
        # each location has 6 faces, we update whether it hits the faces or not
        # face hit but no grid hit is still False because how mask_above is calculated
        # self.coverage_map[gridloc_indices[:, 0], gridloc_indices[:, 1], gridloc_indices[:, 2]] = grid_and_face_hit_mask_flat
        self.coverage_map[
            gridloc_pos_int[..., 0].flatten(),
            gridloc_pos_int[..., 1].flatten(),
            gridloc_pos_int[..., 2].flatten()
        ] = grid_and_face_hit_mask.flatten(0, 1)
        
        # dealing with the below grid voxels
        gridloc_maxdir_moved_by_one = gridloc_pos_int
        gridloc_maxdir_moved_by_one[batch_indices[:, None], :, max_direction_indices[:, None]] -= 1
        assert torch.all(gridloc_maxdir_moved_by_one[:, 1:-1, :] >= 0), f"{gridloc_maxdir_moved_by_one[:, 1:-1, :].min()}"
        
        # gridloc_mov1_indices = einops.rearrange(gridloc_maxdir_moved_by_one[:, 1:-1, :], "num_rays grid_lines_m2 three -> (num_rays grid_lines_m2) three")
        # grid_and_face_hit_mask_flat_mov1 = einops.rearrange(grid_and_face_hit_mask[:, 1:-1, :], "num_rays grid_lines_m2 six -> (num_rays grid_lines_m2) six")
        # self.coverage_map[gridloc_mov1_indices[:, 0], gridloc_mov1_indices[:, 1], gridloc_mov1_indices[:, 2]] = grid_and_face_hit_mask_flat_mov1
        self.coverage_map[
            gridloc_maxdir_moved_by_one[:, 1:-1, 0].flatten(),
            gridloc_maxdir_moved_by_one[:, 1:-1, 1].flatten(),
            gridloc_maxdir_moved_by_one[:, 1:-1, 2].flatten()
        ] = grid_and_face_hit_mask[:, 1:-1, :].flatten(0, 1)
        
        return self.coverage_map

    # def update_coverage_map(
    #     self,
    #     origins,
    #     directions,
    #     depths,
    # ):
        
    #     num_rays = origins.shape[0]
    #     origins = origins.to(self.device)
    #     directions = directions.to(self.device)
    #     depths = depths.to(self.device)
        
    #     # some definition for this function:
    #     # wrapped: the space where the contraction is applied (the 4 x 4 x 4 box)
    #     #     rays travel in a curved line in the wrapped space
    #     # uncontracted: the space where the contraction is not applied (the inf x inf x inf world)
    #     #     rays travel in a straight line in the uncontracted space
        
    #     # the number of lines of the grid is 1 more than the number of voxels
    #     grid_line_num = self.num_voxel + 1
    #     # # vectorized calculation to figure out where the grid lines are
    #     # grid_locations_warpped_1d = torch.linspace(-2, 2, grid_line_num, device=self.device)

    #     # # the -2 and 2 at the end correspond to infinity in the uncontracted space
    #     # # causing computation issues, so we use the middle of the first and last voxel as a proxy
    #     # grid_locations_warpped_1d[0] = grid_locations_warpped_1d[0] + self.half_voxel_size
    #     # grid_locations_warpped_1d[-1] = grid_locations_warpped_1d[-1] - self.half_voxel_size

    #     # # the 3 corresponds to x, y, z
    #     # grid_locations_warpped = torch.zeros((grid_line_num, 3), device=self.device)
    #     # # vectorized calculation to figure out where the grid lines are
    #     # grid_locations_warpped[:, 0] = grid_locations_warpped_1d

    #     # grid_locations_uncontracted = SceneContraction(order=torch.inf).undo_forward(grid_locations_warpped)
    #     # grid_locations_uncontracted_1d = grid_locations_uncontracted[:, 0]
        
    #     # vectorized calculation to figure out where the grid lines are
    #     grid_locations_warpped_1d = torch.linspace(-2, 2, grid_line_num, device=self.device)
    #     # the -2 and 2 at the end correspond to infinity in the uncontracted space
    #     # causing computation issues, so we use the middle of the first and last voxel as a proxy
    #     grid_locations_warpped_1d[0] = grid_locations_warpped_1d[0] + self.half_voxel_size
    #     grid_locations_warpped_1d[-1] = grid_locations_warpped_1d[-1] - self.half_voxel_size
    #     # where the grid lines are in the uncontracted space
    #     grid_locations_uncontracted_1d = SceneContraction(order=torch.inf).undo_forward_1d(
    #         positions=grid_locations_warpped_1d,
    #     )

    #     _, max_direction_indices = torch.max(torch.abs(directions), dim=1)
    #     batch_indices = torch.arange(0, num_rays, device=self.device)

    #     # all_t.shape is [grid_line_num, batch_size]
    #     # it signals at what point in time (can be positive or negative),
    #     # does the ray (# batch_size of them) hit the grid lines (# grid_line_num of them)
    #     all_t = (grid_locations_uncontracted_1d.unsqueeze(1) - \
    #             origins[batch_indices, max_direction_indices].unsqueeze(0)) \
    #             / directions[batch_indices, max_direction_indices].unsqueeze(0)
    #     all_t = all_t.T
        
    #     # calculate where does the ray hits the grid in the uncontracted space (calculate all x, y, z)
    #     # g stands for grid lines, r stands for rays
    #     ray_locations_uncontracted = origins.unsqueeze(-2) + all_t.unsqueeze(-1) * directions.unsqueeze(-2)
    #     # the above calculation is equivalent to the following
    #     # all_x = origins[:, 0] + all_t * directions[:, 0]
    #     # all_y = origins[:, 1] + all_t * directions[:, 1]
    #     # all_z = origins[:, 2] + all_t * directions[:, 2]
    #     # all_t = all_t.T
    #     # all_x = all_x.T
    #     # all_y = all_y.T
    #     # all_z = all_z.T
    #     # ray_locations_uncontracted = torch.stack([all_x, all_y, all_z], dim=-1)
        
    #     # warp the ray locations to the wrapped space
    #     ray_locations_warpped = SceneContraction(order=torch.inf).forward(ray_locations_uncontracted)
        
    #     # neg: negative index, pos: positive index
    #     # this is roughly from -32 to 32 for a grid of 64 voxels
    #     gridloc_neg_to_pos = ray_locations_warpped / self.voxel_size
    #     # this is roughly from 0 to 64 for a grid of 64 voxels
    #     gridloc_pos = gridloc_neg_to_pos + self.num_voxel / 2
    #     # get the integer index, positive
    #     gridloc_pos_int = torch.floor(gridloc_pos).int()
        
    #     # these are the grid voxels (same or below the max direction intersection line)
    #     # end is excluded because the intersection voxel at infinity should not be included
    #     gridloc_pos_int_end_excluded = gridloc_pos_int[:, 1:-1, :]
    #     # moved by 1 in the max direction, to account for the fact that both the voxel below and above the intersection line
    #     # should be filled for the max direction
    #     # consider x moving the fastest, then the voxel below and above x=constant lines should be marked as filled
    #     # gridloc_maxdir_moved_by_one: gridloc_maxdir_pos_int_end_excluded_maxdir_moved_by_one
    #     gridloc_maxdir_moved_by_one = gridloc_pos_int_end_excluded.clone()
    #     gridloc_maxdir_moved_by_one[batch_indices[:, None], :, max_direction_indices[:, None]] -= 1
    #     assert torch.all(gridloc_pos_int >= 0), f"{gridloc_pos_int.min()}"
    #     assert torch.all(gridloc_maxdir_moved_by_one >= 0), f"{gridloc_maxdir_moved_by_one.min()}"
        
    #     if self.do_direction:
    #         dummy = 1
            
    #     dirxfaces = torch.matmul(directions, self.voxel_face_vectors) > 0
    #     grid_location_zero_to_pos_int_repeated = gridloc_pos_int.repeat_interleave(6, dim=1)
    #     faces = torch.arange(6).repeat(gridloc_pos_int.shape[1]).unsqueeze(0).unsqueeze(2).repeat(num_rays, 1, 1).to(self.device)
    #     grid_location_zero_to_pos_int_faces = torch.cat([grid_location_zero_to_pos_int_repeated, faces], dim=2)

    #     moved_by_1_in_max_direction_repeated = gridloc_maxdir_moved_by_one.repeat_interleave(6, dim=1)
    #     faces = torch.arange(6).repeat(gridloc_maxdir_moved_by_one.shape[1]).unsqueeze(0).unsqueeze(2).repeat(num_rays, 1, 1).to(self.device)
    #     moved_by_1_in_max_direction_faces = torch.cat([moved_by_1_in_max_direction_repeated, faces], dim=2)

    #     all_t_repeated = all_t.repeat_interleave(6, dim=1)
    #     dirxfaces_repeated = dirxfaces.repeat(1, gridloc_pos_int.shape[1])

    #     depths_repeated = depths.unsqueeze(1).repeat(1, gridloc_pos_int.shape[1]).repeat(1, 6)

    #     mask_above = ((all_t_repeated > 0) & (all_t_repeated < depths_repeated) & dirxfaces_repeated).unsqueeze(-1).expand_as(grid_location_zero_to_pos_int_faces)
    #     occupied_above = grid_location_zero_to_pos_int_faces[mask_above].view(-1, 4)
        
    #     occupied_above = occupied_above.T
    #     self.coverage_map.index_put_(tuple(occupied_above), torch.tensor(1, dtype=torch.bool), accumulate=False)
        
    #     mask_below = mask_above[:, 6:-6, :]
    #     occupied_below = moved_by_1_in_max_direction_faces[mask_below].view(-1, 4).t()
    #     self.coverage_map.index_put_(tuple(occupied_below), torch.tensor(1, dtype=torch.bool), accumulate=False)
    #     return self.coverage_map

def direction_ray_voxel_intersection(
    origins,
    directions,
    depths,
    voxel_size,
    voxel_grid,
    on_cpu: bool = True,
    ):

    if on_cpu:
        device = "cpu"
    else:
        device = directions.device
    
    origins = origins.to(device)
    directions = directions.to(device)
    depths = depths.to(device)
    
    # voxel_grid is 64x64x64x6 (6 faces of the voxel)
    voxel_face_vectors = torch.tensor([
        [1, 0, 0],  # right
        [-1, 0, 0],  # left
        [0, 1, 0],  # top
        [0, -1, 0],  # bottom
        [0, 0, 1],  # front
        [0, 0, -1],  # back
    ], dtype=directions.dtype, device=device)

    grid_delta = voxel_size
    grid_voxel_num = voxel_grid.shape[0]
    grid_line_num = grid_voxel_num + 1
    grid_locations_warpped_1d = torch.linspace(-2, 2, grid_line_num).to(device)

    # the -2 and 2 at the end correspond to infinity in the uncontracted space
    # causing computation issues, so we use the middle of the last voxel as a proxy
    grid_locations_warpped_1d[0] = grid_locations_warpped_1d[0] + grid_delta / 2
    grid_locations_warpped_1d[-1] = grid_locations_warpped_1d[-1] - grid_delta / 2

    grid_locations_warpped = torch.zeros((grid_line_num, 3)).to(device)
    grid_locations_warpped[:, 0] = grid_locations_warpped_1d

    grid_locations_uncontracted = SceneContraction(order=torch.inf).undo_forward(grid_locations_warpped)
    grid_locations_uncontracted_1d = grid_locations_uncontracted[:, 0]

    _, max_direction_indices = torch.max(torch.abs(directions), dim=1)
    batch_indices = torch.arange(0, origins.shape[0]).to(device)

    # fastest moving direction is one of xyz (recorded by max_direction_indices)
    # calculate where the max_direction hits the grid
    all_t = (grid_locations_uncontracted_1d.unsqueeze(1) - \
            origins[batch_indices, max_direction_indices].unsqueeze(0)) \
            / directions[batch_indices, max_direction_indices].unsqueeze(0)
    
    all_x = origins[:, 0] + all_t * directions[:, 0]
    all_y = origins[:, 1] + all_t * directions[:, 1]
    all_z = origins[:, 2] + all_t * directions[:, 2]
    
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
    dirxfaces = torch.matmul(directions, voxel_face_vectors.T) > 0
    assert torch.all(grid_location_zero_to_pos_int >= 0), f"{grid_location_zero_to_pos_int.min()}"
    assert torch.all(moved_by_1_in_max_direction >= 0), f"{moved_by_1_in_max_direction.min()}"
    
    num_rays = origins.shape[0]
    # all 6 sides of voxel
    grid_location_zero_to_pos_int_repeated = grid_location_zero_to_pos_int.repeat_interleave(6, dim=1)
    faces = torch.arange(6).repeat(grid_location_zero_to_pos_int.shape[1]).unsqueeze(0).unsqueeze(2).repeat(num_rays, 1, 1).to(device)
    grid_location_zero_to_pos_int_faces = torch.cat([grid_location_zero_to_pos_int_repeated, faces], dim=2)

    moved_by_1_in_max_direction_repeated = moved_by_1_in_max_direction.repeat_interleave(6, dim=1)
    faces = torch.arange(6).repeat(moved_by_1_in_max_direction.shape[1]).unsqueeze(0).unsqueeze(2).repeat(num_rays, 1, 1).to(device)
    moved_by_1_in_max_direction_faces = torch.cat([moved_by_1_in_max_direction_repeated, faces], dim=2)

    all_t_repeated = all_t.repeat_interleave(6, dim=1)
    dirxfaces_repeated = dirxfaces.repeat(1, grid_location_zero_to_pos_int.shape[1])

    depths_repeated = depths.unsqueeze(1).repeat(1, grid_location_zero_to_pos_int.shape[1]).repeat(1, 6)

    # fill in grid voxels above the intersection line
    mask_above = ((all_t_repeated > 0) & (all_t_repeated < depths_repeated) & dirxfaces_repeated).unsqueeze(-1).expand_as(grid_location_zero_to_pos_int_faces)
    occupied_above = grid_location_zero_to_pos_int_faces[mask_above].view(-1, 4)
    
    occupied_above = occupied_above.T
    voxel_grid.index_put_(tuple(occupied_above), torch.tensor(1, dtype=torch.bool), accumulate=False)
    
    # fill in grid voxels below the intersection line
    mask_below = mask_above[:, 6:-6, :]
    occupied_below = moved_by_1_in_max_direction_faces[mask_below].view(-1, 4).t()
    voxel_grid.index_put_(tuple(occupied_below), torch.tensor(1, dtype=torch.bool), accumulate=False)

    return voxel_grid 

def get_voxel_grid_positions(voxel_size, voxel_grid):
    grid_delta = voxel_size
    grid_voxel_num = voxel_grid.shape[0]
    grid_line_num = grid_voxel_num + 1
    grid_locations_warpped_1d = torch.linspace(-2, 2, grid_line_num)

    # Orient grid locations to be in the center of the voxel, excluding the last voxel
    grid_locations_warpped_1d = grid_locations_warpped_1d[:-1] + grid_delta / 2

    grid_locations_warpped = torch.zeros((grid_voxel_num, 3))
    grid_locations_warpped[:, 0] = grid_locations_warpped_1d

    grid_locations_uncontracted = SceneContraction(order=torch.inf).undo_forward(grid_locations_warpped)
    grid_locations_uncontracted_1d = grid_locations_uncontracted[:, 0]

    # use meshgrid to get all the grid locations of 64x64x64 voxels
    locs_x, locs_y, locz_z = torch.meshgrid(grid_locations_uncontracted_1d, grid_locations_uncontracted_1d, grid_locations_uncontracted_1d)
    all_grid_locations = torch.stack([locs_x, locs_y, locz_z], dim=-1)

    return all_grid_locations

def main():
    voxel_face_vectors = torch.tensor([
        [1, 0, 0],  # right
        [-1, 0, 0],  # left
        [0, 1, 0],  # top
        [0, -1, 0],  # bottom
        [0, 0, 1],  # front
        [0, 0, -1],  # back
    ], dtype=torch.float32)
    # this is the number of voxels in the grid (not the number of lines)
    grid_voxel_num = 11
    grid_line_num = grid_voxel_num + 1
    
    # the occupancy grid to be filled
    occupancy_grid = torch.zeros((grid_voxel_num, grid_voxel_num, grid_voxel_num, 6), dtype=torch.bool)
    
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
    depths = torch.tensor([20, 20])

    dirxfaces = torch.matmul(direction, voxel_face_vectors.T) > 0
    
    _, max_direction_indices = torch.max(torch.abs(direction), dim=1)
    batch_indices = torch.arange(0, origin.shape[0])
    
    # fastest moving direction is x, calculate where x hits the grid
    all_t = (grid_locations_uncontracted_1d.unsqueeze(1) - \
            origin[batch_indices, max_direction_indices].unsqueeze(0)) \
            / direction[batch_indices, max_direction_indices].unsqueeze(0) # num_rays x (num grid voxels + 1)
    
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
    
    num_rays = origin.shape[0]
    # all 6 sides of voxel
    grid_location_zero_to_pos_int_repeated = grid_location_zero_to_pos_int.repeat_interleave(6, dim=1)
    faces = torch.arange(6).repeat(grid_location_zero_to_pos_int.shape[1]).unsqueeze(0).unsqueeze(2).repeat(num_rays, 1, 1)
    grid_location_zero_to_pos_int_faces = torch.cat([grid_location_zero_to_pos_int_repeated, faces], dim=2)

    moved_by_1_in_max_direction_repeated = moved_by_1_in_max_direction.repeat_interleave(6, dim=1)
    faces = torch.arange(6).repeat(moved_by_1_in_max_direction.shape[1]).unsqueeze(0).unsqueeze(2).repeat(num_rays, 1, 1)
    moved_by_1_in_max_direction_faces = torch.cat([moved_by_1_in_max_direction_repeated, faces], dim=2)

    all_t_repeated = all_t.repeat_interleave(6, dim=1)
    dirxfaces_repeated = dirxfaces.repeat(1, grid_location_zero_to_pos_int.shape[1])

    depths_repeated = depths.unsqueeze(1).repeat(1, grid_location_zero_to_pos_int.shape[1]).repeat(1, 6)

    # fill in grid voxels above the intersection line
    mask_above = ((all_t_repeated > 0) & (all_t_repeated < depths_repeated) & dirxfaces_repeated).unsqueeze(-1).expand_as(grid_location_zero_to_pos_int_faces)
    occupied_above = grid_location_zero_to_pos_int_faces[mask_above].view(-1, 4)
    
    occupied_above = occupied_above.T
    occupancy_grid.index_put_(tuple(occupied_above), torch.tensor(1, dtype=torch.bool), accumulate=False)
    
    # fill in grid voxels below the intersection line
    mask_below = mask_above[:, 6:-6, :]
    occupied_below = moved_by_1_in_max_direction_faces[mask_below].view(-1, 4).t()
    occupancy_grid.index_put_(tuple(occupied_below), torch.tensor(1, dtype=torch.bool), accumulate=False)
    
    # warp the x, y and z back
    # ray_locations_uncontracted = torch.stack([x, y, z], dim=1)
    # ray_locations_warpped = SceneContraction(order=torch.inf).forward(ray_locations_uncontracted)
    
    print(f"{occupancy_grid.sum()=}")
    occupancy_grid_no_faces = occupancy_grid[:, :, :, 0] | \
                                occupancy_grid[:, :, :, 1] | \
                                    occupancy_grid[:, :, :, 2] | \
                                        occupancy_grid[:, :, :, 3] | \
                                            occupancy_grid[:, :, :, 4] | \
                                                occupancy_grid[:, :, :, 5]
    visualize_occupancy_grid(occupancy_grid_no_faces)
    print(f"{occupancy_grid_no_faces.sum()=}")
    dummy = 1


if __name__ == "__main__":
    main()
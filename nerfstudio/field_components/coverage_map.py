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


from typing import Optional, Union

import torch
from torch import Tensor, nn
import einops

from nerfstudio.utils.math import Gaussians
import nerfstudio.models.nerfacto as nerfacto
import nerfstudio.field_components.spatial_distortions as spatial_distortions


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


class DirectionRayVoxelIntersection():
    def __init__(
        self,
        num_voxel: int = 64,
        side_length: float = 4.0,
        device: str = "cpu",
        do_orientation: bool = True,
        ngr_model: Optional[nn.Module] = None,
    ):
        self.num_voxel_1d = num_voxel
        self.num_voxel_3d = num_voxel ** 3
        self.side_length = side_length
        self.voxel_size = side_length / num_voxel
        self.half_voxel_size = 0.5 * self.voxel_size
        self.device = device
        self.do_orientation = do_orientation
        if isinstance(ngr_model, nerfacto.NerfactoModel):
            self.ngr_model = ngr_model
        else:
            self.ngr_model = None
        
        if self.do_orientation:
            self.coverage_map = torch.zeros((num_voxel, num_voxel, num_voxel, 6), dtype=torch.bool, device=self.device)
            self.num_voxel_with_faces_3d = self.num_voxel_3d * 6
        else:
            self.coverage_map = torch.zeros((num_voxel, num_voxel, num_voxel), dtype=torch.bool, device=self.device)
            self.num_voxel_with_faces_3d = self.num_voxel_3d
        
        self.voxel_face_vectors = torch.tensor([
            [1, 0, 0],  # right
            [-1, 0, 0],  # left
            [0, 1, 0],  # top
            [0, -1, 0],  # bottom
            [0, 0, 1],  # front
            [0, 0, -1],  # back
        ], dtype=torch.float32, device=self.device).T
        
        self.high_density_map = None
        if self.ngr_model is not None:
            grid_locations = self.get_map_grid_locations()
            densitys, _ = ngr_model.field.get_density_from_warped_positions(grid_locations)
            occupied = densitys >= 0.5
            self.high_density_map = occupied.squeeze(-1)

    def update_coverage_map(
        self,
        origins,
        directions,
        depths,
    ):
        origins = origins.reshape(-1, 3).to(self.device)
        directions = directions.reshape(-1, 3).to(self.device)
        depths = depths.reshape(-1).to(self.device)
        num_rays = origins.shape[0]
        
        # some definition for this function:
        # wrapped: the space where the contraction is applied (the 4 x 4 x 4 box)
        #     rays travel in a curved line in the wrapped space
        # uncontracted: the space where the contraction is not applied (the inf x inf x inf world)
        #     rays travel in a straight line in the uncontracted space
        
        # the number of lines of the grid is 1 more than the number of voxels
        grid_line_num = self.num_voxel_1d + 1

        # vectorized calculation to figure out where the grid lines are
        grid_locations_warpped_1d = torch.linspace(-2, 2, grid_line_num, device=self.device)
        # the -2 and 2 at the end correspond to infinity in the uncontracted space
        # causing computation issues, so we use the middle of the first and last voxel as a proxy
        grid_locations_warpped_1d[0] = grid_locations_warpped_1d[0] + self.half_voxel_size
        grid_locations_warpped_1d[-1] = grid_locations_warpped_1d[-1] - self.half_voxel_size
        # where the grid lines are in the uncontracted space
        grid_locations_uncontracted_1d = spatial_distortions.SceneContraction(order=torch.inf).undo_forward_1d(
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
        ray_locations_warpped = spatial_distortions.SceneContraction(order=torch.inf).forward(ray_locations_uncontracted)
        
        # neg: negative index, pos: positive index
        # this is roughly from -32 to 32 for a grid of 64 voxels
        gridloc_neg_to_pos = ray_locations_warpped / self.voxel_size
        # this is roughly from 0 to 64 for a grid of 64 voxels
        gridloc_pos = gridloc_neg_to_pos + self.num_voxel_1d / 2
        # get the integer index, positive
        gridloc_pos_int = torch.floor(gridloc_pos).int()
        assert torch.all(gridloc_pos_int >= 0), f"{gridloc_pos_int.min()}"
        
        if self.do_orientation:
            # num_rays x 6, bool
            # for each direction, calculate if the ray hits the (six) faces of the voxel
            dirxfaces = torch.matmul(directions, self.voxel_face_vectors) > 0
            dirxfaces_unsquee = einops.rearrange(dirxfaces, 'num_rays six -> num_rays 1 six')
            
            # prepare the grid hit or no hit to create the mask
            all_t_unsquee = einops.rearrange(all_t, 'num_rays grid_lines -> num_rays grid_lines 1')
            depths_unsquee = einops.rearrange(depths, 'num_rays -> num_rays 1 1')
            
            # combine grid hit or no hit with the direction hit or no hit
            grid_and_face_hit_mask = ((all_t_unsquee > 0) & (all_t_unsquee < depths_unsquee) & dirxfaces_unsquee)
        else:
            depths_unsquee = einops.rearrange(depths, 'num_rays -> num_rays 1')
            grid_and_face_hit_mask = ((all_t > 0) & (all_t < depths_unsquee))
        
        # the grid locations are used for indexing the coverage map
        # each location has 6 faces, we update whether it hits the faces or not
        # face hit but no grid hit is still False because how mask_above is calculated
        self.coverage_map[
            gridloc_pos_int[..., 0].flatten(),
            gridloc_pos_int[..., 1].flatten(),
            gridloc_pos_int[..., 2].flatten()
        ] |= grid_and_face_hit_mask.flatten(0, 1)
        # the above is equivalent to:
        # flatten the ray and grid line to prepare for coverage_map update
        # self.coverage_map[gridloc_indices[:, 0], gridloc_indices[:, 1], gridloc_indices[:, 2]] = grid_and_face_hit_mask_flat
        # gridloc_indices = einops.rearrange(gridloc_pos_int, "num_rays grid_lines three -> (num_rays grid_lines) three")
        # grid_and_face_hit_mask_flat = einops.rearrange(grid_and_face_hit_mask, "num_rays grid_lines six -> (num_rays grid_lines) six")
        
        # dealing with the below grid voxels
        gridloc_maxdir_moved_by_one = gridloc_pos_int
        gridloc_maxdir_moved_by_one[batch_indices[:, None], :, max_direction_indices[:, None]] -= 1
        assert torch.all(gridloc_maxdir_moved_by_one[:, 1:-1, :] >= 0), f"{gridloc_maxdir_moved_by_one[:, 1:-1, :].min()}"
        
        self.coverage_map[
            gridloc_maxdir_moved_by_one[:, 1:-1, 0].flatten(),
            gridloc_maxdir_moved_by_one[:, 1:-1, 1].flatten(),
            gridloc_maxdir_moved_by_one[:, 1:-1, 2].flatten()
        ] |= grid_and_face_hit_mask[:, 1:-1, ...].flatten(0, 1)
        # the above is equivalent to:
        # gridloc_mov1_indices = einops.rearrange(gridloc_maxdir_moved_by_one[:, 1:-1, :], "num_rays grid_lines_m2 three -> (num_rays grid_lines_m2) three")
        # grid_and_face_hit_mask_flat_mov1 = einops.rearrange(grid_and_face_hit_mask[:, 1:-1, :], "num_rays grid_lines_m2 six -> (num_rays grid_lines_m2) six")
        # self.coverage_map[gridloc_mov1_indices[:, 0], gridloc_mov1_indices[:, 1], gridloc_mov1_indices[:, 2]] = grid_and_face_hit_mask_flat_mov1
        
        return self.coverage_map
    
    def get_map_grid_locations(
            self,
            mode: str = "normed",
        ):
        """
        normed: the grid locations are normalized to [0, 1] (the 1 x 1 x 1 box)
        warpped: the grid locations are in the wrapped space (the 4 x 4 x 4 box)
        uncontracted: the grid locations are in the uncontracted space (the inf x inf x inf world)
        """
        assert mode in ["normed", "warpped", "uncontracted"], f"mode {mode} not supported"
        # get the center of all voxels
        grid_locations_warpped = torch.linspace(-2 + self.half_voxel_size, 2 - self.half_voxel_size, self.num_voxel_1d, device=self.device)
        if mode == "normed":
            grid_locations = (grid_locations_warpped + 2) / 4
        elif mode == "warpped":
            grid_locations = grid_locations_warpped
        elif mode == "uncontracted":
            grid_locations = spatial_distortions.SceneContraction(order=torch.inf).undo_forward_1d(grid_locations_warpped)
        else:
            raise NotImplementedError(f"mode {mode} not supported")
        
        # make it 3D
        grid_locations = torch.stack(torch.meshgrid(grid_locations, grid_locations, grid_locations), dim=-1)
        return grid_locations
    
    def get_coverage_map_unoriented(self):
        if self.do_orientation:
            return self.coverage_map.any(dim=-1)
        else:
            return self.coverage_map
    
    # coverage pct: based on ray traverse
    # occupied pct: based on the high density map
    # uncovered pct: not covered by ray traverse and not occupied by high density map
    # overlap pct: covered by ray traverse and occupied by high density map
    
    @property
    def coverage_pct_no_orientation(self):
        if self.do_orientation:
            # reduce the last dimension (any face hit is enough)
            # divided by the number of voxels (multiplication of size(0), size(1), size(2))
            return 100.0 * self.coverage_map.any(dim=-1).sum() / self.num_voxel_3d
        else:
            return 100.0 * self.coverage_map.sum() / self.num_voxel_3d
    
    @property
    def coverage_pct_with_orientation(self):
        return 100.0 * self.coverage_map.sum() / self.coverage_map.numel()
        
    @property
    def occupied_pct(self):
        if self.high_density_map is None:
            return None
        return 100.0 * self.high_density_map.sum() / self.num_voxel_3d
    
    @property
    def covered_pct_no_orientation(self):
        # either covered by ray traverse or occupied by high density map
        if self.high_density_map is None:
            return None
        if self.do_orientation:
            return 100.0 * (self.coverage_map.any(dim=-1) | self.high_density_map).sum() / self.num_voxel_3d
        else:
            return 100.0 * (self.coverage_map | self.high_density_map).sum() / self.num_voxel_3d
    
    @property
    def covered_pct_with_orientation(self):
        if self.high_density_map is None:
            return None
        if self.do_orientation:
            # expand the high_density_map to have orientation
            high_density_map_expanded = self.high_density_map.unsqueeze(-1)
            return 100.0 * (self.coverage_map | high_density_map_expanded).sum() / self.num_voxel_with_faces_3d
        else:
            return 100.0 * (self.coverage_map | self.high_density_map).sum() / self.num_voxel_3d
        
    @property
    def uncovered_pct_no_orientation(self):
        if self.high_density_map is None:
            return None
        if self.do_orientation:
            return 100.0 * ((~self.coverage_map.any(dim=-1)) & (~self.high_density_map)).sum() / self.num_voxel_3d
        else:
            return 100.0 * ((~self.coverage_map) & (~self.high_density_map)).sum() / self.num_voxel_3d
    
    @property
    def uncovered_pct_with_orientation(self):
        if self.high_density_map is None:
            return None
        if self.do_orientation:
            # expand the high_density_map to have orientation
            high_density_map_expanded = self.high_density_map.unsqueeze(-1)
            return 100.0 * ((~self.coverage_map) & (~high_density_map_expanded)).sum() / self.num_voxel_with_faces_3d
        else:
            return 100.0 * ((~self.coverage_map) & (~self.high_density_map)).sum() / self.num_voxel_3d
    
    @property
    def overlap_pct_no_orientation(self):
        if self.high_density_map is None:
            return None
        if self.do_orientation:
            return 100.0 * (self.coverage_map.any(dim=-1) & self.high_density_map).sum() / self.num_voxel_3d
        else:
            return 100.0 * (self.coverage_map & self.high_density_map).sum() / self.num_voxel_3d
    
    @property
    def overlap_pct_with_orientation(self):
        if self.high_density_map is None:
            return None
        if self.do_orientation:
            # expand the high_density_map to have orientation
            high_density_map_expanded = self.high_density_map.unsqueeze(-1)
            return 100.0 * (self.coverage_map & high_density_map_expanded).sum() / self.num_voxel_with_faces_3d
        else:
            return 100.0 * (self.coverage_map & self.high_density_map).sum() / self.num_voxel_3d
    
    def report_stats(self):
        if self.ngr_model is not None:
            print(f"occupied: {self.occupied_pct:.2f}%\n"
                f"ray_coverage_w/wo_ori: {self.coverage_pct_with_orientation:.2f}/{self.coverage_pct_no_orientation:.2f}%\n"
                f"covered_w/wo_ori: {self.covered_pct_with_orientation:.2f}/{self.covered_pct_no_orientation:.2f}%\n"
                f"uncovered_w/wo_ori: {self.uncovered_pct_with_orientation:.2f}/{self.uncovered_pct_no_orientation:.2f}%\n"
                f"overlap_w/wo_ori: {self.overlap_pct_with_orientation:.2f}/{self.overlap_pct_no_orientation:.2f}%")
        else:
            print(f"ray_coverage_w/wo_ori: {self.coverage_pct_with_orientation:.2f}/{self.coverage_pct_no_orientation:.2f}%")


if __name__ == "__main__":
    pass
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

#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import time
import gzip
import json
import os
import shutil
import struct
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
import tyro
import viser.transforms as tf
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated
from simple_parsing.helpers import list_field

from runner_utils.runner_trainers import NerfactoTrainerConfig
from nerfstudio.cameras.camera_paths import get_interpolated_camera_path, get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.scannetpp_dataparser import ScanNetppDataParserConfig
from nerfstudio.data.datasets.base_dataset import Dataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command
from nerfstudio.field_components.coverage_map import DirectionRayVoxelIntersection
import nerfstudio.models.nerfacto as nerfacto


import tables
import pgzip

from nerfstudio.field_components.spatial_distortions import SceneContraction


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    render_nearest_camera=False,
    check_occlusions: bool = False,
    do_coverage_map: bool = False,
    coverage_map_num_voxel: int = 64,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        depth_near_plane: Closest depth to consider when using the colormap for depth. If None, use min value.
        depth_far_plane: Furthest depth to consider when using the colormap for depth. If None, use max value.
        colormap_options: Options for colormap.
        render_nearest_camera: Whether to render the nearest training camera to the rendered camera.
        check_occlusions: If true, checks line-of-sight occlusions when computing camera distance and rejects cameras not visible to each other
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    fps = len(cameras) / seconds

    # JS CODE for Depth based scene coverage
    coverage_grid_class: Optional[DirectionRayVoxelIntersection] = None
    if do_coverage_map:
        coverage_grid_class = DirectionRayVoxelIntersection(
            num_voxel=coverage_map_num_voxel,
            device=cameras.camera_to_worlds.device,
            do_orientation=True,
            ngr_model=pipeline.model,
        )
    
    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)

    with ExitStack() as stack:
        writer = None

        if render_nearest_camera:
            assert pipeline.datamanager.train_dataset is not None
            train_dataset = pipeline.datamanager.train_dataset
            train_cameras = train_dataset.cameras.to(pipeline.device)
        else:
            train_dataset = None
            train_cameras = None

        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                obb_box = None
                if crop_data is not None:
                    obb_box = crop_data.obb

                max_dist, max_idx = -1, -1
                true_max_dist, true_max_idx = -1, -1

                if render_nearest_camera:
                    assert pipeline.datamanager.train_dataset is not None
                    assert train_dataset is not None
                    assert train_cameras is not None
                    cam_pos = cameras[camera_idx].camera_to_worlds[:, 3].cpu()
                    cam_quat = tf.SO3.from_matrix(cameras[camera_idx].camera_to_worlds[:3, :3].numpy(force=True)).wxyz

                    for i in range(len(train_cameras)):
                        train_cam_pos = train_cameras[i].camera_to_worlds[:, 3].cpu()
                        # Make sure the line of sight from rendered cam to training cam is not blocked by any object
                        bundle = RayBundle(
                            origins=cam_pos.view(1, 3),
                            directions=((cam_pos - train_cam_pos) / (cam_pos - train_cam_pos).norm()).view(1, 3),
                            pixel_area=torch.tensor(1).view(1, 1),
                            nears=torch.tensor(0.05).view(1, 1),
                            fars=torch.tensor(100).view(1, 1),
                            camera_indices=torch.tensor(0).view(1, 1),
                            metadata={},
                        ).to(pipeline.device)
                        outputs = pipeline.model.get_outputs(bundle)

                        q = tf.SO3.from_matrix(train_cameras[i].camera_to_worlds[:3, :3].numpy(force=True)).wxyz
                        # calculate distance between two quaternions
                        rot_dist = 1 - np.dot(q, cam_quat) ** 2
                        pos_dist = torch.norm(train_cam_pos - cam_pos)
                        dist = 0.3 * rot_dist + 0.7 * pos_dist

                        if true_max_dist == -1 or dist < true_max_dist:
                            true_max_dist = dist
                            true_max_idx = i

                        if outputs["depth"][0] < torch.norm(cam_pos - train_cam_pos).item():
                            continue

                        if check_occlusions and (max_dist == -1 or dist < max_dist):
                            max_dist = dist
                            max_idx = i

                    if max_idx == -1:
                        max_idx = true_max_idx

                if crop_data is not None:
                    with renderers.background_color_override_context(
                        crop_data.background_color.to(pipeline.device)
                    ), torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            cameras[camera_idx : camera_idx + 1], obb_box=obb_box
                        )
                else:
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            cameras[camera_idx : camera_idx + 1], obb_box=obb_box
                        )
                        cam = cameras[camera_idx]
                        ray_bundle = cam.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)

                        # Get time taken to run dda_ray_traversal_vectorized
                        if do_coverage_map:
                            start = time.time()
                            coverage_grid = coverage_grid_class.update_coverage_map(
                                origins = ray_bundle.origins,
                                directions = ray_bundle.directions,
                                depths = outputs['depth'],
                            )
                            end = time.time()
                            print()
                            print(f"Camera id: {camera_idx}, Time: {end - start:.5f} sec")
                            coverage_grid_class.report_stats()

                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)
                    output_image = outputs[rendered_output_name]
                    is_depth = rendered_output_name.find("depth") != -1
                    if is_depth:
                        output_image = (
                            colormaps.apply_depth_colormap(
                                output_image,
                                accumulation=outputs["accumulation"],
                                near_plane=depth_near_plane,
                                far_plane=depth_far_plane,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    else:
                        output_image = (
                            colormaps.apply_colormap(
                                image=output_image,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    render_image.append(output_image)

                # Add closest training image to the right of the rendered image
                if render_nearest_camera:
                    assert train_dataset is not None
                    assert train_cameras is not None
                    img = train_dataset.get_image_float32(max_idx)
                    height = cameras.image_height[0]
                    # maintain the resolution of the img to calculate the width from the height
                    width = int(img.shape[1] * (height / img.shape[0]))
                    resized_image = torch.nn.functional.interpolate(
                        img.permute(2, 0, 1)[None], size=(int(height), int(width))
                    )[0].permute(1, 2, 0)
                    resized_image = (
                        colormaps.apply_colormap(
                            image=resized_image,
                            colormap_options=colormap_options,
                        )
                        .cpu()
                        .numpy()
                    )
                    render_image.append(resized_image)

                render_image = np.concatenate(render_image, axis=1)
                if output_format == "images":
                    if image_format == "png":
                        media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image, fmt="png")
                    if image_format == "jpeg":
                        media.write_image(
                            output_image_dir / f"{camera_idx:05d}.jpg", render_image, fmt="jpeg", quality=jpeg_quality
                        )
                if output_format == "video":
                    if writer is None:
                        render_width = int(render_image.shape[1])
                        render_height = int(render_image.shape[0])
                        writer = stack.enter_context(
                            media.VideoWriter(
                                path=output_filename,
                                shape=(render_height, render_width),
                                fps=fps,
                            )
                        )
                    writer.add_image(render_image)

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    if output_format == "video":
        if cameras.camera_type[0] == CameraType.EQUIRECTANGULAR.value:
            CONSOLE.print("Adding spherical camera data")
            insert_spherical_metadata_into_file(output_filename)
        table.add_row("Video", str(output_filename))
    else:
        table.add_row("Images", str(output_image_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))

    if isinstance(pipeline.model, NerfactoTrainerConfig):
        # Compute total coverage
        total_coverage = torch.sum(coverage_grid).item()
        total_voxels = (num_voxels ** 3) * 6
        coverage_percentage = (total_coverage / total_voxels) * 100
        print("Total Coverage Percentage:", coverage_percentage)


def insert_spherical_metadata_into_file(
    output_filename: Path,
) -> None:
    """Inserts spherical metadata into MP4 video file in-place.
    Args:
        output_filename: Name of the (input and) output file.
    """
    # NOTE:
    # because we didn't use faststart, the moov atom will be at the end;
    # to insert our metadata, we need to find (skip atoms until we get to) moov.
    # we should have 0x00000020 ftyp, then 0x00000008 free, then variable mdat.
    spherical_uuid = b"\xff\xcc\x82\x63\xf8\x55\x4a\x93\x88\x14\x58\x7a\x02\x52\x1f\xdd"
    spherical_metadata = bytes(
        """<rdf:SphericalVideo
xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
xmlns:GSpherical='http://ns.google.com/videos/1.0/spherical/'>
<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
<GSpherical:Spherical>True</GSpherical:Spherical>
<GSpherical:Stitched>True</GSpherical:Stitched>
<GSpherical:StitchingSoftware>nerfstudio</GSpherical:StitchingSoftware>
</rdf:SphericalVideo>""",
        "utf-8",
    )
    insert_size = len(spherical_metadata) + 8 + 16
    with open(output_filename, mode="r+b") as mp4file:
        try:
            # get file size
            mp4file_size = os.stat(output_filename).st_size

            # find moov container (probably after ftyp, free, mdat)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"moov":
                    break
                mp4file.seek(pos + size)
            # if moov isn't at end, bail
            if pos + size != mp4file_size:
                # TODO: to support faststart, rewrite all stco offsets
                raise Exception("moov container not at end of file")
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # go inside moov
            mp4file.seek(pos + 8)
            # find trak container (probably after mvhd)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"trak":
                    break
                mp4file.seek(pos + size)
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # we need to read everything from end of trak to end of file in order to insert
            # TODO: to support faststart, make more efficient (may load nearly all data)
            mp4file.seek(pos + size)
            rest_of_file = mp4file.read(mp4file_size - pos - size)
            # go to end of trak (again)
            mp4file.seek(pos + size)
            # insert our uuid atom with spherical metadata
            mp4file.write(struct.pack(">I4s16s", insert_size, b"uuid", spherical_uuid))
            mp4file.write(spherical_metadata)
            # write rest of file
            mp4file.write(rest_of_file)
        finally:
            mp4file.close()


@dataclass
class CropData:
    """Data for cropping an image."""

    background_color: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """background color"""
    obb: OrientedBox = field(default_factory=lambda: OrientedBox(R=torch.eye(3), T=torch.zeros(3), S=torch.ones(3) * 2))
    """Oriented box representing the crop region"""

    # properties for backwards-compatibility interface
    @property
    def center(self):
        return self.obb.T

    @property
    def scale(self):
        return self.obb.S


def get_crop_from_json(camera_json: Dict[str, Any]) -> Optional[CropData]:
    """Load crop data from a camera path JSON

    args:
        camera_json: camera path data
    returns:
        Crop data
    """
    if "crop" not in camera_json or camera_json["crop"] is None:
        return None
    bg_color = camera_json["crop"]["crop_bg_color"]
    center = camera_json["crop"]["crop_center"]
    scale = camera_json["crop"]["crop_scale"]
    rot = (0.0, 0.0, 0.0) if "crop_rot" not in camera_json["crop"] else tuple(camera_json["crop"]["crop_rot"])
    assert len(center) == 3
    assert len(scale) == 3
    assert len(rot) == 3
    return CropData(
        background_color=torch.Tensor([bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]),
        obb=OrientedBox.from_params(center, rot, scale),
    )


@dataclass
class BaseRender:
    """Base class for rendering."""

    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("renders/output.mp4")
    """Path to output video file."""
    image_format: Literal["jpeg", "png"] = "jpeg"
    """Image format"""
    jpeg_quality: int = 100
    """JPEG quality"""
    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    depth_near_plane: Optional[float] = None
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = None
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
    """Colormap options."""
    render_nearest_camera: bool = False
    """Whether to render the nearest training camera to the rendered camera."""
    check_occlusions: bool = False
    """If true, checks line-of-sight occlusions when computing camera distance and rejects cameras not visible to each other"""
    
    # from Gavin
    do_coverage_map: bool = False
    """Whether to compute coverage map"""
    coverage_map_num_voxel = 64
    """Number of voxels in coverage map"""


@dataclass
class RenderCameraPath(BaseRender):
    """Render a camera path generated by the viewer or blender add-on."""

    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    def main(self,
             do_coverage_map_override=None) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # temp folder for writing left and right view renders
            temp_folder_path = self.output_path.parent / (self.output_path.stem + "_temp")

            Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
            left_eye_path = temp_folder_path / "render_left.mp4"

            self.output_path = left_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                CONSOLE.print("[bold green]:goggles: Omni-directional Stereo VR :goggles:")
            else:
                CONSOLE.print("[bold green]:goggles: VR180 :goggles:")

            CONSOLE.print("Rendering left eye view")

        # add mp4 suffix to video output if none is specified
        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
            do_coverage_map=self.do_coverage_map if do_coverage_map_override is None else do_coverage_map_override,
            coverage_map_num_voxel=self.coverage_map_num_voxel,
        )

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # declare paths for left and right renders

            left_eye_path = self.output_path
            right_eye_path = left_eye_path.parent / "render_right.mp4"

            self.output_path = right_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                camera_path.camera_type[0] = CameraType.OMNIDIRECTIONALSTEREO_R.value
            else:
                camera_path.camera_type[0] = CameraType.VR180_R.value

            CONSOLE.print("Rendering right eye view")
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                depth_near_plane=self.depth_near_plane,
                depth_far_plane=self.depth_far_plane,
                colormap_options=self.colormap_options,
                render_nearest_camera=self.render_nearest_camera,
                check_occlusions=self.check_occlusions,
            )

            self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_R.value:
                # stack the left and right eye renders vertically for ODS final output
                ffmpeg_ods_command = ""
                if self.output_format == "video":
                    ffmpeg_ods_command = f'ffmpeg -y -i "{left_eye_path}" -i "{right_eye_path}" -filter_complex "[0:v]pad=iw:2*ih[int];[int][1:v]overlay=0:h" -c:v libx264 -crf 23 -preset veryfast "{self.output_path}"'
                    run_command(ffmpeg_ods_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_ods_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final ODS Render Complete")
            else:
                # stack the left and right eye renders horizontally for VR180 final output
                self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")
                ffmpeg_vr180_command = ""
                if self.output_format == "video":
                    ffmpeg_vr180_command = f'ffmpeg -y -i "{right_eye_path}" -i "{left_eye_path}" -filter_complex "[1:v]hstack=inputs=2" -c:a copy "{self.output_path}"'
                    run_command(ffmpeg_vr180_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_vr180_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final VR180 Render Complete")


@dataclass
class RenderInterpolated(BaseRender):
    """Render a trajectory that interpolates between training or eval dataset images."""

    pose_source: Literal["eval", "train"] = "eval"
    """Pose source to render."""
    interpolation_steps: int = 10
    """Number of interpolation steps between eval dataset cameras."""
    order_poses: bool = False
    """Whether to order camera poses by proximity."""
    frame_rate: int = 24
    """Frame rate of the output video."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
        )

        install_checks.check_ffmpeg_installed()

        if self.pose_source == "eval":
            assert pipeline.datamanager.eval_dataset is not None
            cameras = pipeline.datamanager.eval_dataset.cameras
        else:
            assert pipeline.datamanager.train_dataset is not None
            cameras = pipeline.datamanager.train_dataset.cameras

        seconds = self.interpolation_steps * len(cameras) / self.frame_rate
        camera_path = get_interpolated_camera_path(
            cameras=cameras,
            steps=self.interpolation_steps,
            order_poses=self.order_poses,
        )

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
        )


@dataclass
class SpiralRender(BaseRender):
    """Render a spiral trajectory (often not great)."""

    seconds: float = 3.0
    """How long the video should be."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    frame_rate: int = 24
    """Frame rate of the output video (only for interpolate trajectory)."""
    radius: float = 0.1
    """Radius of the spiral."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
        )

        install_checks.check_ffmpeg_installed()

        assert isinstance(
            pipeline.datamanager,
            (
                VanillaDataManager,
                ParallelDataManager,
                RandomCamerasDataManager,
            ),
        )
        steps = int(self.frame_rate * self.seconds)
        camera_start, _ = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
        camera_path = get_spiral_path(camera_start, steps=steps, radius=self.radius)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=self.seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
        )


@contextmanager
def _disable_datamanager_setup(cls):
    """
    Disables setup_train or setup_eval for faster initialization.
    """
    old_setup_train = getattr(cls, "setup_train")
    old_setup_eval = getattr(cls, "setup_eval")
    setattr(cls, "setup_train", lambda *args, **kwargs: None)
    setattr(cls, "setup_eval", lambda *args, **kwargs: None)
    yield cls
    setattr(cls, "setup_train", old_setup_train)
    setattr(cls, "setup_eval", old_setup_eval)


@dataclass
class DatasetRender(BaseRender):
    """Render all images in the dataset."""

    output_path: Path = Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    split: Literal["train", "val", "test", "train+test"] = "test"
    """Split to render."""
    rendered_output_names: Optional[List[str]] = field(default_factory=lambda: None)
    """Name of the renderer outputs to use. rgb, depth, raw-depth, gt-rgb etc. By default all outputs are rendered."""
    idx: List[int] = list_field(0,)
    render_subset: bool = True

    def main(self,
             do_coverage_map_override=None) -> None:
        do_coverage_map = self.do_coverage_map if do_coverage_map_override is None else do_coverage_map_override
        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            data_manager_config.train_num_images_to_sample_from = -1
            data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference"
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))

        for split in self.split.split("+"):
            datamanager: VanillaDataManager
            dataset: Dataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)
            dataloader = FixedIndicesEvalDataloader(
                input_dataset=dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))
            
            # JS CODE for Depth based scene coverage
            coverage_grid_class: Optional[DirectionRayVoxelIntersection] = None
            do_coverage_map = False if not isinstance(pipeline.model, nerfacto.NerfactoModel) else do_coverage_map
            if do_coverage_map:
                coverage_grid_class = DirectionRayVoxelIntersection(
                    num_voxel=self.coverage_map_num_voxel,
                    device=datamanager.device,
                    do_orientation=True,
                    ngr_model=pipeline.model,
                )
            
            with Progress(
                TextColumn(f":movie_camera: Rendering split {split} :movie_camera:"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
            ) as progress:
                for camera_idx, (camera, batch) in enumerate(progress.track(dataloader, total=len(dataset))):

                    def get_image_idx(dataparser_outputs, camera_idx, images_root):
                        import re

                        image_idx = (
                                dataparser_outputs.image_filenames[camera_idx].with_suffix("").relative_to(images_root)
                            )
                        image_idx = re.sub(r"[^0-9]", "", str(image_idx))

                        return int(image_idx)
                    if not isinstance(data_manager_config.dataparser, ScanNetppDataParserConfig):
                        if self.render_subset and not get_image_idx(dataparser_outputs, camera_idx, images_root) in self.idx:
                            continue
                    else:
                        if self.render_subset and not camera_idx in self.idx:
                            continue

                    with torch.no_grad():
                        if do_coverage_map:
                            outputs, camera_ray_bundle = pipeline.model.get_outputs_for_camera(camera, return_ray_bundle=True)
                        else:
                            outputs = pipeline.model.get_outputs_for_camera(camera)

                    if do_coverage_map:
                        start = time.time()
                        coverage_grid = coverage_grid_class.update_coverage_map(
                            origins=camera_ray_bundle.origins,
                            directions=camera_ray_bundle.directions,
                            depths=outputs["depth"],
                        )
                        end = time.time()
                        print()
                        print(f"Camera id: {camera_idx}, Time: {end - start:.5f} sec")
                        coverage_grid_class.report_stats()

                    gt_batch = batch.copy()
                    gt_batch["rgb"] = gt_batch.pop("image")
                    all_outputs = (
                        list(outputs.keys())
                        + [f"raw-{x}" for x in outputs.keys()]
                        + [f"gt-{x}" for x in gt_batch.keys()]
                        + [f"raw-gt-{x}" for x in gt_batch.keys()]
                    )
                    rendered_output_names = self.rendered_output_names
                    if rendered_output_names is None:
                        rendered_output_names = ["gt-rgb"] + list(outputs.keys())
                    for rendered_output_name in rendered_output_names:
                        if rendered_output_name not in all_outputs:
                            CONSOLE.rule("Error", style="red")
                            CONSOLE.print(
                                f"Could not find {rendered_output_name} in the model outputs", justify="center"
                            )
                            CONSOLE.print(
                                f"Please set --rendered-output-name to one of: {all_outputs}", justify="center"
                            )
                            sys.exit(1)

                        is_raw = False
                        is_depth = rendered_output_name.find("depth") != -1
                        image_name = f"{camera_idx:05d}"

                        # Try to get the original filename
                        image_name = (
                            dataparser_outputs.image_filenames[camera_idx].with_suffix("").relative_to(images_root)
                        )

                        # [GAVINCHECK]
                        output_path = self.output_path / split / rendered_output_name / image_name
                        # output_path = self.output_path / split / rendered_output_name / os.path.basename(dataparser_outputs.image_filenames[camera_idx])
                        output_path.parent.mkdir(exist_ok=True, parents=True)

                        output_name = rendered_output_name
                        if output_name.startswith("raw-"):
                            output_name = output_name[4:]
                            is_raw = True
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                                if is_depth:
                                    # Divide by the dataparser scale factor
                                    output_image.div_(dataparser_outputs.dataparser_scale)
                        else:
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                        del output_name

                        # Map to color spaces / numpy
                        if is_raw:
                            output_image = output_image.cpu().numpy()
                        elif is_depth:
                            output_image = (
                                colormaps.apply_depth_colormap(
                                    output_image,
                                    accumulation=outputs["accumulation"],
                                    near_plane=self.depth_near_plane,
                                    far_plane=self.depth_far_plane,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                        else:
                            output_image = (
                                colormaps.apply_colormap(
                                    image=output_image,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )

                        # Save to file
                        if is_raw:
                            with gzip.open(output_path.with_suffix(".npy.gz"), "wb") as f:
                                np.save(f, output_image)
                        elif self.image_format == "png":
                            media.write_image(output_path.with_suffix(".png"), output_image, fmt="png")
                        elif self.image_format == "jpeg":
                            media.write_image(
                                output_path.with_suffix(".jpg"), output_image, fmt="jpeg", quality=self.jpeg_quality
                            )
                        else:
                            raise ValueError(f"Unknown image format {self.image_format}")

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.split.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Render on split {} Complete :tada:[/bold]", expand=False))

@dataclass
class LERFRender(BaseRender):
    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("renders")
    """Path to output video file."""
    split: Literal["train", "test", "train+test"] = "test"
    """Split to render."""
    rendered_output_names: Optional[List[str]] = field(default_factory=lambda: None)
    """Name of the renderer outputs to use. rgb, depth, raw-depth, raw-clip_embeds. By default all outputs are rendered."""
    idx = [0, 1, 2, 3, 4, 5, 9]
    """Idx of the images to render (defaults to the first 10 images)"""
    render_subset: bool = True
    """Whether to render a subset of the images"""
    avg_pool: bool = False
    """Whether to average pool the clip_embeds"""
    avg_pool_kernel_size: int = 3
    """Kernel size for average pooling"""
    # faiss_index_mode: str = None
    # """Faiss index mode from the database"""
    trained_index_fp: Path = None
    """Location of the trained index"""
    clip_scales: List[float] = field(default_factory=lambda: [1.0])
    """Scales to use for the clip embeds"""
    train_faiss_during_render: bool = False
    """Whether to train the faiss index during render, instead of using pre-trained"""
    faiss_index_mode: str = None
    """Faiss index mode from the database"""

    @staticmethod
    def _get_image_idx(filename, images_root):
        import re

        image_idx = filename.with_suffix("").relative_to(images_root)
        image_idx = re.sub(r"[^0-9]", "", str(image_idx))

        return int(image_idx)

    def main(self):
        
        import faiss
        if "faiss-clip_embeds" in self.rendered_output_names:
            if self.train_faiss_during_render:
                self._faiss_index = faiss.index_factory(512, self.faiss_index_mode, faiss.METRIC_INNER_PRODUCT)
                self._faiss_index_trained = False
            else:
                self._faiss_index = faiss.read_index(self.trained_index_fp)
                self._faiss_index_trained = True
            self._faiss_idx_count = 0 
            self._faiss_image_to_idx_dict = {}
        else:
            self.train_faiss_during_render = False
            
        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference"
        )

        data_manager_config = config.pipeline.datamanager

        for split in self.split.split("+"):
            datamanager: VanillaDataManager
            dataset: Dataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(
                        test_mode="test",
                        device=pipeline.device,
                        image_encoder=pipeline.config.network.setup() # Setup image encoder
                    )

                dataset = datamanager.train_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(
                        test_mode=split, 
                        device=pipeline.device,
                        image_encoder=pipeline.config.network.setup() # Setup image encoder
                    )

                dataset = datamanager.eval_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)

            images_root = Path(dataparser_outputs.image_filenames[0]).parent
            if self.render_subset:
                subset_image_indices = [
                    idx
                    for idx, filename in enumerate(dataparser_outputs.image_filenames) 
                    if (self._get_image_idx(filename, images_root) - 1) in self.idx # Subtract 1 as LERF files begins counting from 1
                ]

            dataloader = FixedIndicesEvalDataloader(
                input_dataset=dataset,
                image_indices=subset_image_indices if self.render_subset else None,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            with Progress(
                TextColumn(f":movie_camera: Rendering split {split} :movie_camera:"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
                TextColumn(text_format="{task.description}"),
            ) as progress:
                for camera_idx, (camera, batch) in enumerate(progress.track(dataloader, total=len(dataloader.image_indices))):

                    image_idx = batch['image_idx']
                    progress.update(task_id=0, description=f"(LERF: currently rendering {image_idx=}...)")

                    with torch.inference_mode():
                        pipeline.model.nerfret_inference_mode = True
                        pipeline.model.nerfret_best_scales = self.clip_scales
                        pipeline.model.nerfret_outputs = ["clip_embeds", "rgb"]
                        outputs = pipeline.model.get_outputs_for_camera(camera)

                        torch.cuda.empty_cache()

                    if self.avg_pool:
                        outputs["clip_embeds"] = torch.nn.functional.avg_pool2d(outputs["clip_embeds"].permute(2, 0, 1), self.avg_pool_kernel_size).permute(1, 2, 0)

                    progress.update(task_id=0, description=f"(LERF: finished rendering {image_idx=}...)")

                    # # Convert to float16. Does not work with PQ indexes
                    # outputs["clip_embeds"] = outputs["clip_embeds"].to(torch.float16)

                    # Use the first render as training embeddings
                    if self.train_faiss_during_render and not self._faiss_index_trained:
                        progress.update(task_id=0, description=f"(LERF: Training faiss index. Using {image_idx=} as training embeddings...)")
                        train_embs = outputs["clip_embeds"].cpu().numpy()
                        train_embs = train_embs.reshape(-1, train_embs.shape[-1])
                        faiss.normalize_L2(train_embs)
                        self._faiss_index.train(train_embs)

                        self._faiss_index_trained = True

                    VALID_OUTPUTS = (
                        list(outputs.keys())
                        + [f"raw-{x}" for x in outputs.keys()]
                        + ["faiss-clip_embeds"]
                    )
                    # Make sure we don't try to render clip_embeds as an image
                    if "clip_embeds" in VALID_OUTPUTS:
                        VALID_OUTPUTS.remove("clip_embeds")

                    rendered_output_names = self.rendered_output_names
                    if rendered_output_names is None:
                        # Render everything if nothing was specified.
                        rendered_output_names = VALID_OUTPUTS

                    for rendered_output_name in rendered_output_names:
                        if rendered_output_name not in VALID_OUTPUTS:
                            CONSOLE.rule("Error", style="red")
                            CONSOLE.print(
                                f"Could not find {rendered_output_name} in the model outputs", justify="center"
                            )
                            CONSOLE.print(
                                f"Please set --rendered-output-name to one of: {VALID_OUTPUTS}", justify="center"
                            )
                            sys.exit(1)

                        # Special handling for faiss-clip_embeds
                        if rendered_output_name == "faiss-clip_embeds":
                            progress.update(task_id=0, description=f"(LERF: now adding {rendered_output_name} of {image_idx=} to faiss index)")

                            emb = outputs["clip_embeds"].cpu().numpy()
                            emb = emb.reshape(-1, emb.shape[-1])
                            faiss.normalize_L2(emb)
                            self._faiss_index.add_with_ids(
                                emb, np.arange(self._faiss_idx_count, self._faiss_idx_count + emb.shape[0])
                            )
                            self._faiss_idx_count += emb.shape[0]

                            # Use the output path of the RGB image as the key
                            image_name = (
                                dataparser_outputs.image_filenames[image_idx].with_suffix("").relative_to(images_root)
                            )
                            output_path_str = os.path.join(self.output_path, split, "rgb", image_name)
                            output_path = Path(output_path_str)

                            # org_idx = self._get_image_idx(dataparser_outputs.image_filenames[image_idx], images_root) - 1 # Subtract 1 as LERF files begins counting from 1
                            self._faiss_image_to_idx_dict[output_path_str] = self._faiss_idx_count
                        else:
                            is_raw = False
                            is_depth = rendered_output_name.find("depth") != -1
                            # image_name = f"{image_idx:05d}"

                            # Try to get the original filename
                            image_name = (
                                dataparser_outputs.image_filenames[image_idx].with_suffix("").relative_to(images_root)
                            )

                            # [GAVINCHECK]
                            output_path_str = os.path.join(self.output_path, split, rendered_output_name, image_name)
                            output_path = Path(output_path_str)
                            output_path.parent.mkdir(exist_ok=True, parents=True)

                            output_name = rendered_output_name
                            if output_name.startswith("raw-"):
                                output_name = output_name[4:]
                                is_raw = True
                                output_image = outputs[output_name]
                                if is_depth:
                                    # Divide by the dataparser scale factor
                                    output_image.div_(dataparser_outputs.dataparser_scale)
                            else:
                                output_image = outputs[output_name]
                            del output_name

                            # Map to color spaces / numpy
                            if is_raw:
                                output_image = output_image.cpu().numpy()
                            elif is_depth:
                                output_image = (
                                    colormaps.apply_depth_colormap(
                                        output_image,
                                        accumulation=outputs["accumulation"],
                                        near_plane=self.depth_near_plane,
                                        far_plane=self.depth_far_plane,
                                        colormap_options=self.colormap_options,
                                    )
                                    .cpu()
                                    .numpy()
                                )
                            else:
                                output_image = (
                                    colormaps.apply_colormap(
                                        image=output_image,
                                        colormap_options=self.colormap_options,
                                    )
                                    .cpu()
                                    .numpy()
                                )

                            # TODO: I've seen the image_idx and actual filename to be off by quite a bit in some cases 
                            # (like idx=103, filename=frame_00106 - these are not exact, I don't remember them, but I have seen off by 3)
                            # Perhaps use a debug log or print a warning if the image_idx and filename are off by more than 1

                            progress.update(task_id=0, description=f"(LERF: now saving {rendered_output_name} rendering of {image_idx=} to {output_path})")
                            # Save to file
                            if is_raw:
                                _threads = max(len(os.sched_getaffinity(0)) // 2, 1) # Use half of the available cores
                                _blocksize = output_image.nbytes // (_threads * 2) # Divide work into 2 blocks per thread (fastest setting, though I only tested two divisors: _threads and _threads * 2)
                                # print(output_image.nbytes, _threads, _blocksize)
                                with pgzip.open(output_path.with_suffix(".npy.gz"), "wb", thread=_threads, blocksize=_blocksize, compresslevel=1) as f:
                                    np.save(f, output_image)
                                # Old solutions, no longer needed (hopefully):
                                # import lzma
                                # with lzma.open(output_path.with_suffix(".npy.lzma"), "wb") as f:
                                #     np.save(f, output_image)
                                #
                                # Doing compression takes too long and does not save us much disk space (if at all). 
                                # Using the native numpy.savez_compressed seems to be fastest for now at about same filesize
                                # np.savez_compressed(output_path, data=output_image)
                                #
                                # import tables
                                # with tables.open_file(str(output_path.with_suffix('.h5')), mode='w') as f:
                                #     atom = tables.Float64Atom()
                                #     array_c = f.create_carray(f.root, 'data', atom, output_image.shape, filters=tables.Filters(complevel=1, complib='lzo'))
                                #     array_c[:] = output_image
                            elif self.image_format == "png":
                                media.write_image(output_path.with_suffix(".png"), output_image, fmt="png")
                            elif self.image_format == "jpeg":
                                media.write_image(
                                    output_path.with_suffix(".jpg"), output_image, fmt="jpeg", quality=self.jpeg_quality
                                )
                            else:
                                raise ValueError(f"Unknown image format {self.image_format}")
                    
                    del outputs
                    del output_image

        if "faiss-clip_embeds" in self.rendered_output_names:
            faiss.write_index(
                self._faiss_index,
                os.path.join(self.output_path, "faiss_scene.index")
            )
            with open(os.path.join(self.output_path, "faiss_image_to_idx_dict.json"), "w", encoding="utf-8") as f:
                json.dump(self._faiss_image_to_idx_dict, f)

            del self._faiss_index
            del self._faiss_idx_count
            del self._faiss_image_to_idx_dict

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.split.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Render on split {} Complete :tada:[/bold]", expand=False))

    
@dataclass
class LERFRenderCameraPath(BaseRender):
    """Render a camera path generated by the viewer or blender add-on."""

    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_path: Path = None
    """Path to output video file."""
    rendered_output_names: Optional[List[str]] = field(default_factory=None)
    """Name of the renderer outputs to use. rgb, depth, raw-depth, raw-clip_embeds. By default all outputs are rendered."""
    avg_pool: bool = False
    """Whether to average pool the clip_embeds"""
    avg_pool_kernel_size: int = 3
    """Kernel size for average pooling"""
    trained_index_fp: Path = None
    """Location of the trained faiss index"""
    clip_scales: List[float] = field(default_factory=lambda: [1.0])
    """Scales to use for the clip_embeds"""
    train_faiss_during_render: bool = False
    """Whether to train the faiss index during render, instead of using pre-trained"""
    faiss_index_mode: str = None
    """Faiss index mode from the database"""

    def main(self) -> None:
        """Main function."""
        assert self.output_path is not None

        import faiss
        if "faiss-clip_embeds" in self.rendered_output_names:
            if self.train_faiss_during_render:
                self._faiss_index_is_trained = False
                self._faiss_index = faiss.index_factory(512, self.faiss_index_mode, faiss.METRIC_INNER_PRODUCT)
            else:
                self._faiss_index = faiss.read_index(self.trained_index_fp)
                self._faiss_index_is_trained = True
            self._faiss_idx_count = 0 
            self._faiss_image_to_idx_dict = {}

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        # seconds = camera_path["seconds"]
        # crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        data_manager_config = config.pipeline.datamanager
        with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
            datamanager = data_manager_config.setup(
                test_mode="test", 
                device=pipeline.device,
                image_encoder=pipeline.config.network.setup() # Setup image encoder
            )

        dataset = datamanager.eval_dataset
        dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
        if dataparser_outputs is None:
            dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)

        with Progress(
            TextColumn(f":movie_camera: Rendering camera paths for LERF :movie_camera:"),
            BarColumn(),
            TaskProgressColumn(
                text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                show_speed=True,
            ),
            ItersPerSecColumn(suffix="fps"),
            TimeRemainingColumn(elapsed_when_finished=False, compact=False),
            TimeElapsedColumn(),
            TextColumn(text_format="{task.description}"),
        ) as progress:
            for camera_idx, camera in enumerate(progress.track(camera_path, total=len(camera_path))):
                with torch.inference_mode():
                    pipeline.model.nerfret_inference_mode = True
                    pipeline.model.nerfret_best_scales = self.clip_scales
                    pipeline.model.nerfret_outputs = ["clip_embeds", "rgb"]
                    outputs = pipeline.model.get_outputs_for_camera(camera)

                    torch.cuda.empty_cache()

                if self.avg_pool:
                    outputs["clip_embeds"] = torch.nn.functional.avg_pool2d(outputs["clip_embeds"].permute(2, 0, 1), self.avg_pool_kernel_size).permute(1, 2, 0)

                # Use the first render as training embeddings
                if self.train_faiss_during_render and self._faiss_index_is_trained is False:
                    train_embs = outputs["clip_embeds"].cpu().numpy()
                    train_embs = train_embs.reshape(-1, train_embs.shape[-1])

                    faiss.normalize_L2(train_embs)
                    self._faiss_index.train(train_embs)

                    self._faiss_index_is_trained = True

                VALID_OUTPUTS = (
                    list(outputs.keys())
                    + [f"raw-{x}" for x in outputs.keys()]
                )
                # Make sure we don't try to render clip_embeds as an image
                if "clip_embeds" in VALID_OUTPUTS:
                    VALID_OUTPUTS.remove("clip_embeds")

                rendered_output_names = self.rendered_output_names
                if rendered_output_names is None:
                    # Render everything if nothing was specified.
                    rendered_output_names = VALID_OUTPUTS

                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in VALID_OUTPUTS:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(
                            f"Could not find {rendered_output_name} in the model outputs", justify="center"
                        )
                        CONSOLE.print(
                            f"Please set --rendered-output-name to one of: {VALID_OUTPUTS}", justify="center"
                        )
                        sys.exit(1)

                    if rendered_output_name == "faiss-clip_embeds":
                        progress.update(task_id=0, description=f"(LERF: now adding {rendered_output_name} of {camera_idx=} to faiss index)")

                        emb = outputs["clip_embeds"].cpu().numpy()
                        emb = emb.reshape(-1, emb.shape[-1])
                        faiss.normalize_L2(emb)
                        self._faiss_index.add_with_ids(
                            emb, np.arange(self._faiss_idx_count, self._faiss_idx_count + emb.shape[0])
                        )
                        self._faiss_idx_count += emb.shape[0]

                        # Use the output path of the RGB image as the key
                        output_path = self.output_path / "rgb" / camera_idx

                        # org_idx = self._get_image_idx(dataparser_outputs.image_filenames[image_idx], images_root) - 1 # Subtract 1 as LERF files begins counting from 1
                        self._faiss_image_to_idx_dict[str(output_path)] = self._faiss_idx_count
                    else:
                        is_raw = False
                        is_depth = rendered_output_name.find("depth") != -1
                        image_name = f"{camera_idx:05d}"

                        # [GAVINCHECK]
                        output_path = os.path.join(self.output_path, "camera_path", rendered_output_name, image_name)
                        output_path.parent.mkdir(exist_ok=True, parents=True)

                        output_name = rendered_output_name
                        if output_name.startswith("raw-"):
                            output_name = output_name[4:]
                            is_raw = True
                            output_image = outputs[output_name]
                            if is_depth:
                                # Divide by the dataparser scale factor
                                output_image.div_(dataparser_outputs.dataparser_scale)
                        else:
                            output_image = outputs[output_name]
                        del output_name

                        # Map to color spaces / numpy
                        if is_raw:
                            output_image = output_image.cpu().numpy()
                        elif is_depth:
                            output_image = (
                                colormaps.apply_depth_colormap(
                                    output_image,
                                    accumulation=outputs["accumulation"],
                                    near_plane=self.depth_near_plane,
                                    far_plane=self.depth_far_plane,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                        else:
                            output_image = (
                                colormaps.apply_colormap(
                                    image=output_image,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )

                        # Save to file
                        if is_raw:
                            _threads = max(len(os.sched_getaffinity(0)) // 2, 1) # Use half of the available cores
                            _blocksize = output_image.nbytes // (_threads * 2) # Divide work into 2 blocks per thread (fastest setting, though I only tested two divisors: _threads and _threads * 2)
                            print(output_image.nbytes, _threads, _blocksize)
                            with pgzip.open(output_path.with_suffix(".npy.gz"), "wb", thread=_threads, blocksize=_blocksize, compresslevel=1) as f:
                                np.save(f, output_image)
                        elif self.image_format == "png":
                            media.write_image(output_path.with_suffix(".png"), output_image, fmt="png")
                        elif self.image_format == "jpeg":
                            media.write_image(
                                output_path.with_suffix(".jpg"), output_image, fmt="jpeg", quality=self.jpeg_quality
                            )
                        else:
                            raise ValueError(f"Unknown image format {self.image_format}")
                
                del outputs
                del output_image

        if "faiss-clip_embeds" in self.rendered_output_names:
            faiss.write_index(
                self._faiss_index,
                os.path.join(self.output_path, "faiss_scene.index")
            )
            with open(os.path.join(self.output_path, "faiss_image_to_idx_dict.json"), "w", encoding="utf-8") as f:
                json.dump(self._faiss_image_to_idx_dict, f)

            del self._faiss_index
            del self._faiss_idx_count
            del self._faiss_image_to_idx_dict


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RenderInterpolated, tyro.conf.subcommand(name="interpolate")],
        Annotated[SpiralRender, tyro.conf.subcommand(name="spiral")],
        Annotated[DatasetRender, tyro.conf.subcommand(name="dataset")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa

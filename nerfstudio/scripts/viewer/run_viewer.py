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
Starts viewer in eval mode.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

import tyro

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer as ViewerState
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState
from nerfstudio.cameras.camera_paths import get_path_from_json
import json



@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""

    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewer:
    """Load a checkpoint and start the viewer."""

    load_config: Path
    """Path to config YAML file."""
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    """Viewer configuration"""
    vis: Literal["viewer", "viewer_legacy"] = "viewer"
    """Type of viewer"""

    use_cameras_from_json: bool=False
    """Whether to use cameras from JSON file"""
    
    cameras1: str=""
    cameras2: str=""
    color1: str=""
    color2: str=""
    """ Camera JSON file paths and colors for cameras 1 and 2"""

    def main(self) -> None:
        """Main function."""
        config, pipeline, _, step = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config.vis = self.vis
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk

        camera_json_path = []
        colors = []
        if self.use_cameras_from_json:
            if self.cameras1:
                camera_json_path.append(self.cameras1)
                colors.append(self.color1)
            if self.cameras2:
                camera_json_path.append(self.cameras2)
                colors.append(self.color2)
        _start_viewer(config, pipeline, step, self.use_cameras_from_json, camera_json_path, colors)

    def save_checkpoint(self, *args, **kwargs):
        """
        Mock method because we pass this instance to viewer_state.update_scene
        """


def _start_viewer(config: TrainerConfig, pipeline: Pipeline, step: int, use_cameras_from_json: bool = False, camera_json_paths: list = [], colors: list = []):
    """Starts the viewer

    Args:
        config: Configuration of pipeline to load
        pipeline: Pipeline instance of which to load weights
        step: Step at which the pipeline was saved
    """
    base_dir = config.get_base_dir()
    viewer_log_path = base_dir / config.viewer.relative_log_filename
    banner_messages = None
    viewer_state = None
    if config.vis == "viewer_legacy":
        viewer_state = ViewerLegacyState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
        )
        banner_messages = [f"Legacy viewer at: {viewer_state.viewer_url}"]
    if config.vis == "viewer":
        viewer_state = ViewerState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            share=config.viewer.make_share_url,
        )
        banner_messages = viewer_state.viewer_info

    # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
    config.logging.local_writer.enable = False
    writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)

    assert viewer_state and pipeline.datamanager.train_dataset

    cameras_list = []
    colorsrgb = []
    if use_cameras_from_json:
        if len(camera_json_paths) == 0:
            raise ValueError("camera_json_paths is empty")
        # Load JSON file into dictionary
        for path in camera_json_paths:
            camera_path_json = json.load(open(path))
            # Convert dictionary to Cameras instance
            cameras = get_path_from_json(camera_path_json)
            cameras_list.append(cameras)
        for color in colors:
            colorsrgb.append(tuple(map(int, color.split(','))))

    viewer_state.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
        cameras_list=cameras_list,
        colors=colorsrgb,
    )
    if isinstance(viewer_state, ViewerLegacyState):
        viewer_state.viser_server.set_training_state("completed")
    viewer_state.update_scene(step=step)
    while True:
        time.sleep(0.01)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[RunViewer]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(tyro.conf.FlagConversionOff[RunViewer])  # noqa

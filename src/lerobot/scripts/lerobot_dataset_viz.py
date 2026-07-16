#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

    - Visualize data stored on a local machine:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize all features of a local custom dataset:
```
local$ lerobot-dataset-viz \
    --repo-id edge_sink_wipe \
    --root dependencies/comet/data/edge_sink_wipe \
    --episode-index 0
```

- Visualize only a subset of features:
```
local$ lerobot-dataset-viz \
    --repo-id edge_sink_wipe \
    --root dependencies/comet/data/edge_sink_wipe \
    --episode-index 0 \
    --keys observation.ft action.virtual_target_position action.contact_direction
```

- Exclude features from visualization:
```
local$ lerobot-dataset-viz \
    --repo-id edge_sink_wipe \
    --root dependencies/comet/data/edge_sink_wipe \
    --episode-index 0 \
    --exclude-keys observation.camera_time.front_camera observation.camera_time.wrist_camera
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --grpc-port 9876

local$ rerun rerun+http://IP:GRPC_PORT/proxy
```

"""

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

# Meta keys used for indexing / Rerun timelines; skipped unless explicitly listed in --keys.
META_KEYS = frozenset(
    {
        "index",
        "frame_index",
        "timestamp",
        "episode_index",
        "task_index",
        "task",
    }
)
IMAGE_DTYPES = frozenset({"image", "video"})
NUMERIC_DTYPES = frozenset({"float32", "float64", "int8", "int16", "int32", "int64", "bool", "uint8"})


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def _dim_label(names: list | dict | None, dim_idx: int) -> str:
    """Resolve a human-readable label for vector dimension ``dim_idx``."""
    if names is None:
        return str(dim_idx)
    if isinstance(names, dict):
        flat: list = []
        for value in names.values():
            if isinstance(value, list):
                flat.extend(value)
            else:
                flat.append(value)
        if dim_idx < len(flat) and flat[dim_idx] is not None:
            return str(flat[dim_idx])
        return str(dim_idx)
    if isinstance(names, list) and dim_idx < len(names) and names[dim_idx] is not None:
        return str(names[dim_idx])
    return str(dim_idx)


def resolve_keys_to_log(
    dataset: LeRobotDataset,
    keys: list[str] | None = None,
    exclude_keys: list[str] | None = None,
) -> list[str]:
    """Select which dataset features to log to Rerun.

    By default logs every feature except meta/index keys. ``keys`` restricts to an
    explicit allow-list (meta keys are allowed if named). ``exclude_keys`` removes
    keys from the selection.
    """
    features = dataset.meta.features
    available = set(features)

    if keys is not None:
        missing = [key for key in keys if key not in available]
        if missing:
            raise ValueError(
                f"Unknown feature key(s): {missing}. Available: {sorted(available)}"
            )
        selected = list(keys)
    else:
        selected = [key for key in features if key not in META_KEYS]

    if exclude_keys:
        exclude_set = set(exclude_keys)
        unknown_exclude = exclude_set - available
        if unknown_exclude:
            logging.warning(f"Ignoring unknown --exclude-keys: {sorted(unknown_exclude)}")
        selected = [key for key in selected if key not in exclude_set]

    return selected


def log_sample_feature(
    key: str,
    value: torch.Tensor | np.ndarray | float | int | bool,
    feature_info: dict,
    display_compressed_images: bool = False,
) -> None:
    """Log a single sample feature (image or numeric vector) to Rerun."""
    dtype = feature_info.get("dtype")

    if dtype in IMAGE_DTYPES:
        img = to_hwc_uint8_numpy(value)
        img_entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
        rr.log(key, entity=img_entity)
        return

    if dtype not in NUMERIC_DTYPES:
        logging.debug(f"Skipping non-numeric feature '{key}' with dtype '{dtype}'")
        return

    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        values = [v.item() for v in flat]
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    else:
        values = [value]

    names = feature_info.get("names")
    if len(values) == 1:
        # Scalar features: log under the feature key directly.
        label = _dim_label(names, 0) if names else None
        entity_path = f"{key}/{label}" if label and label != "0" else key
        rr.log(entity_path, rr.Scalars(float(values[0])))
        return

    for dim_idx, val in enumerate(values):
        rr.log(f"{key}/{_dim_label(names, dim_idx)}", rr.Scalars(float(val)))


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    grpc_port: int = 9876,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    keys: list[str] | None = None,
    exclude_keys: list[str] | None = None,
    **kwargs,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id
    keys_to_log = resolve_keys_to_log(dataset, keys=keys, exclude_keys=exclude_keys)
    logging.info(f"Logging {len(keys_to_log)} feature(s): {keys_to_log}")

    logging.info("Loading dataloader")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        logging.info(f"Connect to a Rerun Server: rerun rerun+http://IP:{grpc_port}/proxy")
        rr.serve_web_viewer(open_browser=False, web_port=web_port, connect_to=server_uri)

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["frame_index"][i].item())
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            for key in keys_to_log:
                if key not in batch:
                    continue
                log_sample_feature(
                    key,
                    batch[key][i],
                    dataset.meta.features[key],
                    display_compressed_images=display_compressed_images,
                )

    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun rerun+http://IP:GRPC_PORT/proxy` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        help="deprecated, please use --grpc-port instead.",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=9876,
        help="gRPC port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="If set, display compressed images in Rerun instead of uncompressed ones.",
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional allow-list of feature keys to visualize. "
            "By default, all features except meta/index keys are logged. "
            "Example: --keys observation.ft action.contact_direction"
        ),
    )
    parser.add_argument(
        "--exclude-keys",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional feature keys to exclude from visualization. "
            "Example: --exclude-keys observation.camera_time.front_camera"
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")
    episode_index = kwargs.pop("episode_index")

    if kwargs["ws_port"] is not None:
        logging.warning(
            "--ws-port is deprecated and will be removed in future versions. Please use --grpc-port instead."
        )
        logging.warning("Setting grpc_port to ws_port value.")
        kwargs["grpc_port"] = kwargs.pop("ws_port")
    else:
        kwargs.pop("ws_port")

    init_logging()
    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[episode_index], root=root, tolerance_s=tolerance_s)

    visualize_dataset(dataset, episode_index=episode_index, **kwargs)


if __name__ == "__main__":
    main()

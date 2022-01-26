"""Module with loading dataset"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from cv2 import cv2
from tqdm import tqdm

from cots_dataset.synthetic.utils import make_blend_mask, norm_color_transfer
from cots_dataset.utils import (
    load_dataframe,
    prepare_annotations,
    preprocess_annotations,
)


def _separate_annotations(
    annotations: Dict[str, Tuple[Path, List[List[int]]]]
) -> Tuple[Dict[str, Tuple[Path, List[List[int]]]], Dict[str, Path]]:
    annotations_cots = {}
    annotations_empty = {}

    for image_id, image_info in annotations.items():
        if len(image_info[1]) == 0:
            annotations_empty[image_id] = image_info[0]
        else:
            annotations_cots[image_id] = image_info

    return annotations_cots, annotations_empty


def assign_cots_annotations_to_empty(
    annotations_cots: Dict[str, Tuple[Path, List[List[int]]]],
    annotations_empty: Dict[str, Path],
) -> List[Tuple[Path, Tuple[Path, List[List[int]]]]]:
    cots_values = list(annotations_cots.values())
    empty_values = list(annotations_empty.values())

    if len(empty_values) < len(cots_values):
        raise ValueError('Empty images are less, than cots images')

    iter_times = len(empty_values) // len(cots_values)
    residual_times = len(empty_values) % len(cots_values)
    cots_values_additional = cots_values * iter_times + cots_values[:residual_times]

    random.shuffle(cots_values_additional)

    assigned_values = list(zip(empty_values, cots_values_additional))

    return assigned_values


def cut_out_cots(
    image: np.ndarray, cots_boxes: List[List[int]], border: float = 0.3
) -> List[Tuple[np.ndarray, List[int]]]:
    all_crops_and_coords = []

    for box in cots_boxes:
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]

        x_border = int((box[3] - box[1]) * border)
        y_border = int((box[2] - box[0]) * border)

        x_start = int(box[1]) - x_border
        x_start = x_start if x_start > 0 else 0
        x_end = int(box[3]) + x_border
        x_end = x_end if x_end < image.shape[0] else image.shape[0]

        y_start = int(box[0]) - y_border
        y_start = y_start if y_start > 0 else 0
        y_end = int(box[2]) + y_border
        y_end = y_end if y_end < image.shape[1] else image.shape[1]

        crop = image[
            x_start:x_end,
            y_start:y_end,
            :,
        ]
        object_coords = [
            box[0] - y_start,
            box[1] - x_start,
            box[3] - box[1],
            box[2] - box[0],
        ]

        all_crops_and_coords.append((crop, object_coords))

    return all_crops_and_coords


def merge_images(
    assigned_values: List[Tuple[Path, Tuple[Path, List[List[int]]]]]
) -> None:
    for empty_image_path, (cots_image_path, cots_boxes) in tqdm(assigned_values):
        empty_image = cv2.imread(str(empty_image_path))
        cots_image = cv2.imread(str(cots_image_path))

        all_crops_and_coords = cut_out_cots(image=cots_image, cots_boxes=cots_boxes)

        for curr_crop_and_coord in all_crops_and_coords:
            crop, coord = curr_crop_and_coord

            cots_image_width, cots_image_height, _ = cots_image.shape
            crop_width, crop_height, _ = crop.shape

            blend_mask = make_blend_mask(
                size=(crop_height, crop_width), object_box=coord
            )
            blend_mask = blend_mask * 0.9

            x_min = random.randint(0, cots_image_width - crop_width)
            y_min = random.randint(0, cots_image_height - crop_height)
            new_box = [
                coord[0] + x_min,
                coord[1] + y_min,
                coord[2],
                coord[3],
            ]

            empty_image_crop = empty_image[
                x_min : x_min + crop_width, y_min : y_min + crop_height
            ]
            crop = norm_color_transfer(crop, empty_image_crop)

            mix_image = (1 - blend_mask) * empty_image_crop + blend_mask * crop

            empty_image[
                x_min : x_min + crop_width, y_min : y_min + crop_height
            ] = mix_image
            cv2.rectangle(
                empty_image,
                (new_box[1], new_box[0]),
                (new_box[1] + new_box[3], new_box[0] + new_box[2]),
                (0, 0, 255),
                2,
            )

        root_images = Path('res')
        root_images.mkdir(exist_ok=True)
        new_image_path = root_images.joinpath(empty_image_path.name)
        cv2.imwrite(str(new_image_path), empty_image)


if __name__ == '__main__':
    _dataframe_path = '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/train.csv'
    _images_folder = Path(
        '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/train_images'
    )

    _dataframe = load_dataframe(path=_dataframe_path, video=1)

    _annotations = prepare_annotations(dataframe=_dataframe)
    _annotations_preprocessed = preprocess_annotations(
        images_folder=_images_folder, annotations=_annotations
    )

    _annotations_cots, _annotations_empty = _separate_annotations(
        annotations=_annotations_preprocessed
    )

    _assigned_values = assign_cots_annotations_to_empty(
        annotations_cots=_annotations_cots, annotations_empty=_annotations_empty
    )

    merge_images(assigned_values=_assigned_values)

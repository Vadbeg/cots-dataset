"""Module with dataset building using crops"""


from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from cv2 import cv2
from tqdm import tqdm

from cots_dataset.utils import (
    draw_boxes,
    load_dataframe,
    prepare_annotations,
    preprocess_annotations,
)


class DatasetBuilder:
    ANNOTATIONS_COLUMN = 'annotations'
    VIDEO_ID_COLUMN = 'video_id'
    IMAGE_ID_COLUMN = 'image_id'
    VIDEO_FRAME_COLUMN = 'video_frame'

    def __init__(
        self,
        images_root: Path,
        crop_images_root: Path,
        annotations_path: Path,
        crop_size: Tuple[int, int] = (420, 420),
        max_shift: int = 40,
    ):
        self.images_root: Path = images_root
        self.crop_images_root: Path = crop_images_root
        self.annotations = load_dataframe(path=annotations_path, video=2)

        self.crop_size = crop_size
        self.max_shift = max_shift

    def build(self):
        annotations = prepare_annotations(dataframe=self.annotations)
        annotations = preprocess_annotations(
            images_folder=self.images_root, annotations=annotations
        )

        all_annotations = []

        for index, value in tqdm(annotations.items(), postfix='Preparing crops...'):
            image_path, boxes = value
            image = cv2.imread(str(image_path))

            crops_and_boxes = self._build_one_image(image=image, boxes=boxes)
            image_crops_annotations = self._save_one_image_crops(
                crops_and_boxes=crops_and_boxes, original_image_path=image_path
            )
            all_annotations.extend(image_crops_annotations)

        dataframe = pd.json_normalize(data=all_annotations)
        dataframe.to_csv('res.csv', index=False)

    def _save_one_image_crops(
        self,
        crops_and_boxes: List[Tuple[np.ndarray, List[List[int]]]],
        original_image_path: Path,
    ) -> List[Dict[str, Any]]:
        crop_parent_path = self.crop_images_root.joinpath(
            original_image_path.parent.name
        )
        crop_parent_path.mkdir(parents=True, exist_ok=True)
        crops_annotations = []

        for idx, (crop, boxes) in enumerate(crops_and_boxes):
            crop_filename = original_image_path.with_stem(
                f'{original_image_path.stem}_{idx}'
            ).name
            crop_path = crop_parent_path.joinpath(crop_filename)
            cv2.imwrite(str(crop_path), crop)

            video_id = int(crop_path.parent.name.split('_')[-1])
            image_id = '.'.join(crop_filename.split('.')[:-1])
            crop_id = f'{video_id}-{image_id}'

            curr_annotation = {
                self.IMAGE_ID_COLUMN: crop_id,
                self.VIDEO_ID_COLUMN: video_id,
                self.ANNOTATIONS_COLUMN: self._prepare_annotations(boxes=boxes),
            }
            crops_annotations.append(curr_annotation)

        return crops_annotations

    @staticmethod
    def _prepare_annotations(boxes: List[List[int]]) -> List[Dict[str, Any]]:
        annotations = []

        for curr_box in boxes:
            curr_annotation = {
                'x': curr_box[0],
                'y': curr_box[1],
                'width': curr_box[2],
                'height': curr_box[3],
            }
            annotations.append(curr_annotation)

        return annotations

    def _build_one_image(
        self, image: np.ndarray, boxes: List[List[int]]
    ) -> List[Tuple[np.ndarray, List[List[int]]]]:
        height, width, _ = image.shape

        image_to_draw = draw_boxes(image=image, boxes=boxes)
        cv2.imshow('Image', image_to_draw)

        crops_and_boxes = []

        for curr_x in range(0, width, self.crop_size[0]):
            if curr_x > width - self.crop_size[0]:
                if width - curr_x > self.max_shift:
                    curr_x = width - self.crop_size[0]
                else:
                    continue

            for curr_y in range(0, height, self.crop_size[1]):
                if curr_y > height - self.crop_size[1]:
                    if width - curr_x > self.max_shift:
                        curr_y = height - self.crop_size[1]
                    else:
                        continue

                curr_crop = image[
                    curr_y : curr_y + self.crop_size[1],
                    curr_x : curr_x + self.crop_size[0],
                ]
                crop_boxes = self._get_boxes_for_coords(
                    boxes=boxes,
                    min_x=curr_x,
                    min_y=curr_y,
                    max_x=curr_x + self.crop_size[0],
                    max_y=curr_y + self.crop_size[1],
                )

                crops_and_boxes.append((curr_crop, crop_boxes))

        return crops_and_boxes

    @staticmethod
    def _get_boxes_for_coords(
        boxes: List[List[int]],
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
        min_box_percent: float = 0.3,
    ) -> List[List[int]]:
        new_boxes = []

        for curr_box in boxes:
            box_min_x, box_min_y, box_width, box_height = curr_box
            original_box_width, original_box_height = box_width, box_height

            if box_min_x > max_x:
                continue
            if box_min_y > max_y:
                continue

            if box_min_x < min_x < box_min_x + box_width:
                box_width = box_width - (min_x - box_min_x)
                box_min_x = min_x
            elif box_min_x < min_x:
                continue
            if box_min_y < min_y < box_min_y + box_height:
                box_height = box_height - (min_y - box_min_y)
                box_min_y = min_y
            elif box_min_y < min_y:
                continue

            new_box_width = max_x - box_min_x
            new_box_width = new_box_width if new_box_width < box_width else box_width
            new_box_height = max_y - box_min_y
            new_box_height = (
                new_box_height if new_box_height < box_height else box_height
            )

            if (new_box_width / original_box_width) * (
                new_box_height / original_box_height
            ) < min_box_percent:
                continue

            curr_new_box = [
                box_min_x - min_x,
                box_min_y - min_y,
                new_box_width,
                new_box_height,
            ]
            new_boxes.append(curr_new_box)

        return new_boxes

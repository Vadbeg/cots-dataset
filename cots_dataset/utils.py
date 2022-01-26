"""Module with top level utils"""


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from cv2 import cv2


def _preprocess_dataframe(annotations_dataframe: pd.DataFrame) -> pd.DataFrame:
    annotations_column = 'annotations'
    video_id_column = 'video_id'
    video_frame_column = 'video_frame'

    if isinstance(annotations_dataframe[annotations_column].iloc[0], str):
        annotations_dataframe[annotations_column] = annotations_dataframe[
            annotations_column
        ].apply(eval)

    annotations_dataframe[video_id_column] = annotations_dataframe[
        video_id_column
    ].apply(int)
    annotations_dataframe[video_frame_column] = annotations_dataframe[
        video_frame_column
    ].apply(int)
    annotations_dataframe['num_bbox'] = annotations_dataframe[annotations_column].apply(
        lambda x: len(x)
    )

    return annotations_dataframe


def load_dataframe(path: Union[Path, str], video: Optional[int] = None) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    if video:
        dataframe = dataframe.loc[dataframe['video_id'] == video]

    dataframe = _preprocess_dataframe(annotations_dataframe=dataframe)

    return dataframe


def _get_image_path_by_id(images_folder: Path, image_id: str) -> Path:
    video_id, frame_id = image_id.split('-')
    image_path = images_folder.joinpath(f'video_{video_id}', f'{frame_id}.jpg')

    return image_path


def _preprocess_boxes(ann_boxes: List[Dict[str, int]]) -> List[List[int]]:
    boxes = []

    for curr_box in ann_boxes:
        boxes.append(
            [curr_box['x'], curr_box['y'], curr_box['width'], curr_box['height']]
        )

    return boxes


def preprocess_annotations(
    images_folder: Path, annotations: List[Dict[str, Any]]
) -> Dict[str, Tuple[Path, List[List[int]]]]:
    preprocessed_annotations = dict()

    for curr_annotation in annotations:
        image_id = curr_annotation['image_id']

        image_path = _get_image_path_by_id(
            images_folder=images_folder, image_id=image_id
        )
        boxes = _preprocess_boxes(ann_boxes=curr_annotation['annotations'])

        preprocessed_annotations[image_id] = (image_path, boxes)

    return preprocessed_annotations


def prepare_annotations(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    annotations = (
        dataframe.groupby('image_id')
        .apply(lambda x: x.to_json(orient='records'))
        .tolist()
    )
    annotations = list(map(lambda x: eval(x)[0], annotations))

    return annotations


def draw_boxes(image: np.ndarray, boxes: List[List[int]]) -> np.ndarray:
    image_draw = np.copy(image)

    for curr_box in boxes:
        cv2.rectangle(
            image_draw,
            (curr_box[0], curr_box[1]),
            (curr_box[0] + curr_box[2], curr_box[1] + curr_box[3]),
            (0, 0, 255),
            2,
        )

    return image_draw

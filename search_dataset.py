import time
from pathlib import Path

from cv2 import cv2
from tqdm import tqdm

from cots_dataset.dataset import (
    load_dataframe,
    prepare_annotations,
    preprocess_annotations,
)

if __name__ == '__main__':
    _dataframe_path = '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/train.csv'
    _images_folder = Path(
        '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/train_images'
    )

    _dataframe = load_dataframe(path=_dataframe_path, video=1)

    _annotations = prepare_annotations(dataframe=_dataframe)
    _annotations = preprocess_annotations(
        images_folder=_images_folder, annotations=_annotations
    )
    image_and_boxes = list(_annotations.values())
    image_and_boxes = sorted(
        image_and_boxes, key=lambda x: (x[0].parent, int(x[0].name.split('.')[0]))
    )

    idx = 0
    stop = False

    prog_bar = tqdm(image_and_boxes)
    while idx < len(image_and_boxes):
        curr_image_path, curr_boxes = image_and_boxes[idx]
        image = cv2.imread(str(curr_image_path))

        time.sleep(0.001)
        for box in curr_boxes:
            cv2.rectangle(
                image,
                (box[0], box[1]),
                (box[0] + box[2], box[1] + box[3]),
                (0, 0, 255),
                2,
            )

        prog_bar.n = idx
        prog_bar.refresh()

        cv2.imshow('Image', image)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            idx = idx - 30 if idx > 30 else 0
        elif key == ord('d'):
            idx = idx + 30 if idx < len(image_and_boxes) - 30 else len(image_and_boxes)
        elif key == ord('s'):
            stop = False if stop else True
        elif key == ord('o'):
            idx = idx - 1 if idx > 1 else 0
        elif key == ord('p'):
            idx = idx + 1 if idx < len(image_and_boxes) - 1 else len(image_and_boxes)

        if not stop:
            idx += 1

    cv2.destroyWindow('Image')

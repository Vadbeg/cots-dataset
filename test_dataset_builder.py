"""Module with testing of dataset builder"""


from pathlib import Path

from cots_dataset.crop.dataset_building import DatasetBuilder

if __name__ == '__main__':
    images_root = Path(
        '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/train_images'
    )
    new_images_root = Path(
        '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/crop_images'
    )
    annotations_path = Path(
        '/Users/vadim.tsitko/Data/tensorflow-great-barrier-reef/train.csv'
    )

    dataset_builder = DatasetBuilder(
        images_root=images_root,
        crop_images_root=new_images_root,
        annotations_path=annotations_path,
    )
    dataset_builder.build()

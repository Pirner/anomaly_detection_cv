from typing import List

import cv2
from torch.utils.data import Dataset


class ImagePathDataset(Dataset):
    """
    Dataset which is responsible for loading data from image paths.
    Each image path given during creation of the dataset will be accessed
    """
    def __init__(self, im_paths: List[str], transforms=None):
        """
        -
        :param im_paths: path to all the images which are part of the dataset and shall be loaded
        :param transforms: transformations to apply to each image path when accessing it
        """
        self.data = im_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        access an item from the dataset with an id given
        :param item:
        :return:
        """
        im_path = self.data[item]
        im = cv2.imread(im_path)
        sample = im

        if self.transforms:
            sample = self.transforms(image=im)

        return sample['image']

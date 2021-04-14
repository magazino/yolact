import cv2
import json
import numpy as np
import os
import os.path as osp
import random
import sys
import torch
import torch.nn.functional as F
import torch.utils.data as data
from .config import cfg
from pycocotools import mask as maskUtils
from skimage import draw


class BoxSegmentationDataset(data.Dataset):
    """Magazino box segmentation dataset"""

    def __init__(self, dataset_name, image_path, info_file):
        self.name = dataset_name
        self.image_info = []
        self.load_dataset(image_path, info_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.image_info)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        image_info = self.image_info[index]
        h, w = image_info['height'], image_info['width']
        target = []
        label = 1  # label for the box class
        masks = np.zeros([len(image_info['polygons']), image_info['height'],
            image_info['width']], dtype=np.uint8)
        for idx, polygon in enumerate(image_info['polygons']):
            x_coords = polygon['all_points_x']
            y_coords = polygon['all_points_y']
            xmin, xmax = np.min(x_coords), np.max(x_coords)
            ymin, ymax = np.min(y_coords), np.max(y_coords)
            target.append([xmin/w, ymin/h, xmax/w, ymax/h, label])
            rr, cc = draw.polygon(y_coords, x_coords)
            masks[idx, rr, cc] = 1
        image = cv2.imread(image_info['path'])
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.float() / 255.0
        num_crowds = 0
        return image, target, masks, h, w, num_crowds

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def load_dataset(self, image_dir, annotation_file):
        """ Loads dataset for box segmentation
        Args:
            image_dir: directory containing the images
            annotation_file: name of the annotation file
        """
        print("Annotation file path: ", annotation_file)
        assert os.path.exists(annotation_file)
        annotations = json.load(open(annotation_file))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(image_dir, a['filename'])
            image = cv2.imread(image_path)[..., ::-1]
            height, width = image.shape[:2]
            self.add_image(
                "box",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

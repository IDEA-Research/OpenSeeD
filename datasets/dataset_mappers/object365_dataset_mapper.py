# Copyright (c) IDEA 2023, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from fvcore.common.config import CfgNode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.structures import (
    Boxes,
    Instances,
)

from ...Networks.Mask2Former.utils import configurable
from ...Networks.Mask2Former.utils import box_ops


"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    min_scale = cfg_input['MIN_SCALE']
    max_scale = cfg_input['MAX_SCALE']

    augmentation = []

    if cfg_input['RANDOM_FLIP'] != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                vertical=cfg_input['RANDOM_FLIP'] == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])
    
    return augmentation

class Object365Mapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        aug_cfg = CfgNode({'INPUT': cfg['INPUT']})
        augs = utils.build_augmentation(aug_cfg, is_train)

        if cfg['INPUT']['CROP']['ENABLED'] and is_train:
            augs.insert(0, T.RandomCrop(cfg['INPUT']['CROP']['TYPE'], cfg['INPUT']['CROP']['SIZE']))
            recompute_boxes = cfg['MODEL']['MASK_ON']
        else:
            recompute_boxes = False

        # augs = build_transform_gen(cfg, is_train)
        # recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg['INPUT']['FORMAT'],
            "use_instance_mask": cfg['MODEL']['MASK_ON'],
            "instance_mask_format": cfg['INPUT']['MASK_FORMAT'],
            "use_keypoint": cfg['MODEL']['KEYPOINT_ON'],
            "recompute_boxes": recompute_boxes,
        }

        if cfg['MODEL']['KEYPOINT_ON']:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg['DATASETS']['TRAIN'])
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        bbox = np.array(dataset_dict['bbox'])
        bbox = box_ops.box_xywh_to_xyxy(torch.as_tensor(bbox)).numpy()
        
        aug_input = T.AugInput(image, boxes=bbox)
        transforms = self.augmentations(aug_input)
        image, bbox_gt = aug_input.image, aug_input.boxes
        
        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        instances = Instances(image_shape)
        instances.gt_boxes = Boxes(torch.as_tensor(bbox_gt))
        
        classes = dataset_dict['categories']
        # HACK object365 class id start from 1.
        classes = torch.tensor(classes, dtype=torch.int64) - 1
        instances.gt_classes = classes
        
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

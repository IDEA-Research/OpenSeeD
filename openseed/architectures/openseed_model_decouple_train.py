# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MaskDINO https://github.com/IDEA-Research/MaskDINO by Hao Zhang and Feng Li.
# ------------------------------------------------------------------------
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .registry import register_model
from ..utils import configurable, box_ops, get_class_names
from ..backbone import build_backbone, Backbone
from ..body import build_openseed_head
from ..modules import sem_seg_postprocess, HungarianMatcher, SetCriterion
from ..language import build_language_encoder

from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
import random
import json
class OpenSeeD(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
        train_dataset_name: str,
        background: bool,
        coco_on=True,
        coco_mask_on=True,
        o365_on=True,
        criterion_coco=None,
        criterion_o365=None,
        split_panno=False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.split_panno=split_panno
        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        self.train_class_names = dict()
        self.train_dataset_name = train_dataset_name
        self.coco_mask_on = coco_mask_on
        self.task_switch = {'coco': coco_on, 'o365': o365_on}
        self.criterion_coco=criterion_coco
        self.criterion_o365=criterion_o365

        print("self.task_switch ", self.task_switch)
        # HACK for only two datasets for seg and det
        if coco_on:
            task = 'seg'
            if not coco_mask_on:
                task = 'det'
            self.train_class_names[task] = get_class_names(train_dataset_name[0], background=background)
            self.train_class_names[task] = [a.replace("-merged", "").replace("-other", "").replace("-stuff", "") for a
                                             in self.train_class_names[task]]
            train_class_names = []
            for name in self.train_class_names[task]:
                names = name.split('-')
                if len(names) > 1:
                    assert len(names) == 2
                    train_class_names.append(names[1] + ' ' + names[0])
                else:
                    train_class_names.append(name)
            self.train_class_names[task] = train_class_names

        if o365_on and len(train_dataset_name)>1:
            for dt in train_dataset_name:
                if "o365" in train_dataset_name or "object365" in train_dataset_name:
                    break
            self.train_class_names['det'] = get_class_names(dt, background=background)
            self.train_class_names['det'] = [a.lower().split('/') for a in self.train_class_names['det']]

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        # Loss parameters:
        deep_supervision = dec_cfg['DEEP_SUPERVISION']
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']

        # loss weights
        class_weight = dec_cfg['CLASS_WEIGHT']
        cost_class_weight = dec_cfg['COST_CLASS_WEIGHT']
        cost_dice_weight = dec_cfg['COST_DICE_WEIGHT']
        dice_weight = dec_cfg['DICE_WEIGHT']
        cost_mask_weight = dec_cfg['COST_MASK_WEIGHT']
        mask_weight = dec_cfg['MASK_WEIGHT']
        cost_box_weight = dec_cfg['COST_BOX_WEIGHT']
        box_weight = dec_cfg['BOX_WEIGHT']
        cost_giou_weight = dec_cfg['COST_GIOU_WEIGHT']
        giou_weight = dec_cfg['GIOU_WEIGHT']

        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
        )

        # MaskDINO losses and weight_dict
        weight_dict = {"loss_mask_cls_0": class_weight}
        weight_dict.update({"loss_mask_bce_0": mask_weight, "loss_mask_dice_0": dice_weight})
        weight_dict.update({"loss_bbox_0":box_weight,"loss_giou_0":giou_weight})
        # two stage is the query selection scheme
        if dec_cfg['TWO_STAGE']:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training
        dn = dec_cfg['DN']
        # TODO hack for dn lable loss
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["dn_labels", "boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["labels", "masks", "boxes"]
        else:
            dn_losses=[]
        if deep_supervision:
            dec_layers = dec_cfg['DEC_LAYERS']
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k.replace('_0', '_{}'.format(i+1)): v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if dec_cfg['BOX']:
            losses = ["labels", "masks","boxes"]
        else:
            losses = ["labels", "masks"]

        # update task switch
        task_switch = {}
        task_switch.update({'bbox': dec_cfg.get('DETECTION', True), 'mask': dec_cfg.get('MASK', True)})
        top_x_layers = {'mask': dec_cfg.get('TOP_MASK_LAYERS', 10),
                        'box': dec_cfg.get('TOP_DETECTION_LAYERS', 10)}

        # building criterion
        criterion_coco = SetCriterion(
            enc_cfg['NUM_CLASSES'],
            matcher=matcher,
            weight_dict=weight_dict,
            top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            grounding_weight=None,
            dn=dec_cfg['DN'],
            dn_losses=dn_losses,
            panoptic_on=dec_cfg['PANO_BOX_LOSS'],
            semantic_ce_loss=dec_cfg['TEST']['SEMANTIC_ON'] and dec_cfg['SEMANTIC_CE_LOSS'] and not dec_cfg['TEST']['PANOPTIC_ON'],
        )

        criterion_o365 = SetCriterion(
            enc_cfg.get('NUM_CLASSES_O365', 365),
            matcher=matcher,
            weight_dict=weight_dict,
            top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            grounding_weight=None,
            dn=dec_cfg['DN'],
            dn_losses=dn_losses,
            panoptic_on=dec_cfg['PANO_BOX_LOSS'],
            semantic_ce_loss=dec_cfg['TEST']['SEMANTIC_ON'] and dec_cfg['SEMANTIC_CE_LOSS'] and not dec_cfg['TEST'][
                'PANOPTIC_ON'],
        )

        # build model
        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)
        lang_encoder = build_language_encoder(cfg)
        sem_seg_head = build_openseed_head(cfg, backbone.output_shape(), lang_encoder, extra=extra)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion_coco": criterion_coco,
            "criterion_o365": criterion_o365,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['COCO']['TEST']['DETECTIONS_PER_IMAGE'],
            "data_loader": None,
            "focus_on_box": cfg['MODEL']['DECODER']['TEST']['TEST_FOUCUS_ON_BOX'],
            "transform_eval": cfg['MODEL']['DECODER']['TEST']['PANO_TRANSFORM_EVAL'],
            "pano_temp": cfg['MODEL']['DECODER']['TEST']['PANO_TEMPERATURE'],
            "semantic_ce_loss": cfg['MODEL']['DECODER']['TEST']['SEMANTIC_ON'] and cfg['MODEL']['DECODER']['SEMANTIC_CE_LOSS'] and not cfg['MODEL']['DECODER']['TEST']['PANOPTIC_ON'],
            "train_dataset_name": cfg['DATASETS']['TRAIN'], # HACK for only two training set
            "background": cfg['MODEL'].get('BACKGROUND', True),
            "coco_on": dec_cfg.get('COCO', True),
            "coco_mask_on": dec_cfg.get('COCO_MASK', True),
            "o365_on": dec_cfg.get('O365', True),
            "split_panno": dec_cfg.get('PANO_CRITERION', True),

        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, inference_task='seg'):
        # import ipdb; ipdb.set_trace()
        if self.training:
            losses = {}
            if self.task_switch['coco'] and 'coco' in batched_inputs:
                self.criterion_coco.num_classes = 133 if 'pano' in self.train_dataset_name[0] else 80
                # self.criterion.num_classes = 133
                task = 'seg'
                if not self.coco_mask_on:
                    task = 'det'
                # import ipdb; ipdb.set_trace()
                losses_coco = self.forward_seg(batched_inputs['coco'], task=task)
                new_losses_coco = {}
                for key, value in losses_coco.items():
                    new_losses_coco['coco.'+str(key)] = losses_coco[key]
                losses.update(new_losses_coco)
            if self.task_switch['o365'] and 'o365' in batched_inputs:
                self.criterion_o365.num_classes = 365
                losses_o365 = self.forward_seg(batched_inputs['o365'], task='det')
                new_losses_o365 = {}
                for key, value in losses_o365.items():
                    new_losses_o365['o365.'+str(key)] = losses_o365[key]
                losses.update(new_losses_o365)
            return losses
        else:
            processed_results = self.forward_seg(batched_inputs, task=inference_task)
            return processed_results

    def forward_seg(self, batched_inputs, task='seg'):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        if self.training:
            if task == "det" and self.task_switch['o365']:
                train_class_names = [random.sample(name, 1)[0] for name in self.train_class_names['det']]
            else:
                train_class_names = self.train_class_names[task]
            self.sem_seg_head.predictor.lang_encoder.get_text_embeddings(train_class_names, is_eval=False)

            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images, task=task)
            else:
                targets = None
            outputs, mask_dict = self.sem_seg_head(features, targets=targets, task=task)
            # bipartite matching-based loss
            if task=='det':
                criterion=self.criterion_o365
                losses = self.criterion_o365(outputs, targets, mask_dict, task=task)
            else:
                criterion=self.criterion_coco
                losses = self.criterion_coco(outputs, targets, mask_dict, task=task)
            # else
            for k in list(losses.keys()):
                if k in criterion.weight_dict:
                    losses[k] *= criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs, _ = self.sem_seg_head(features)
            mask_cls_results = outputs["pred_logits"]
            mask_box_results = outputs["pred_boxes"]
            if 'seg' in task:
                if task == 'seg':
                    self.semantic_on = self.panoptic_on = self.sem_seg_postprocess_before_inference = self.instance_on = True
                if task == 'pan_seg':
                    self.semantic_on = self.instance_on = False
                    self.panoptic_on = True
                    self.sem_seg_postprocess_before_inference = True
                if task == 'inst_seg':
                    self.semantic_on = self.panoptic_on = False
                    self.instance_on = True
                    self.sem_seg_postprocess_before_inference = True
                if task == 'sem_pan_seg':
                    self.semantic_on = self.panoptic_on = True
                    self.instance_on = False
                    self.sem_seg_postprocess_before_inference = True
                if task == 'inst_pan_seg':
                    self.instance_on = self.panoptic_on = True
                    self.semantic_on = False
                    self.sem_seg_postprocess_before_inference = True
                if task == 'sem_seg':
                    self.instance_on = self.panoptic_on = False
                    self.semantic_on = True
                    self.sem_seg_postprocess_before_inference = True
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

            else:
                self.semantic_on = self.panoptic_on = self.sem_seg_postprocess_before_inference = False
                self.instance_on = True
                mask_pred_results = torch.zeros(mask_box_results.shape[0], mask_box_results.shape[1],2, 2).to(mask_box_results)

            del outputs

            processed_results = []

            for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                new_size = (images.tensor.shape[-2], images.tensor.shape[-1])  # padded size (divisible to 32)


                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference

                if self.instance_on:
                    mask_box_result = mask_box_result.to(mask_pred_result)
                    height = new_size[0]/image_size[0]*height
                    width = new_size[1]/image_size[1]*width
                    mask_box_result = self.box_postprocess(mask_box_result, height, width)
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result[:self.sem_seg_head.predictor.num_queries_test],
                        mask_pred_result[:self.sem_seg_head.predictor.num_queries_test],
                        mask_box_result[:self.sem_seg_head.predictor.num_queries_test], True)
                    # instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                    processed_results[-1]["instances"] = instance_r
            del mask_pred_results
            return processed_results

    def prepare_targets(self, targets, images, task='seg'):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            if task != 'det':
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            else:
                padded_masks = None
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes":box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result,split_anno):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        if split_anno:
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
                self.sem_seg_head.predictor.num_queries_test, 1).flatten(0, 1)
        else:
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                # print(i, len(keep), lab)
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        if self.sem_seg_postprocess_before_inference:
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        else:
            mask_scores_per_image = 1.0
            # labels_per_image = labels_per_image + 1  # HACK for o365 classification
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes

@register_model
def get_segmentation_model(cfg, **kwargs):
    return OpenSeeD(cfg)
# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------------
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .registry import register_body
from .encoder import build_encoder
from .decoder import build_decoder
from ..utils import configurable


class OpenSeeDHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_predictor: nn.Module,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], lang_encoder: nn.Module, extra: dict):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']
        transformer_predictor_in_channels = enc_cfg['CONVS_DIM']

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in enc_cfg['IN_FEATURES']
            },
            "ignore_value": enc_cfg['IGNORE_VALUE'],
            "num_classes": enc_cfg.get('NUM_CLASSES', None),
            "pixel_decoder": build_encoder(cfg, input_shape),
            "loss_weight": enc_cfg['LOSS_WEIGHT'],
            "transformer_predictor": build_decoder(
                cfg,
                transformer_predictor_in_channels,
                lang_encoder,
                mask_classification=True,
                extra=extra,
            ),
        }

    def forward(self, features, mask=None, targets=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        return self.layers(features, mask, targets=targets, target_queries=target_queries, target_vlp=target_vlp, task=task, extra=extra)

    def layers(self, features, mask=None,targets=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features, mask)
        if task == 'teacher':
            predictions = self.predictor.forward_teacher(multi_scale_features, mask_features, mask, targets=targets,
                                                         target_queries=target_queries, target_vlp=target_vlp,
                                                         task=task, extra=extra)
        else:
            predictions = self.predictor(multi_scale_features, mask_features, mask, targets=targets,
                                         target_queries=target_queries, target_vlp=target_vlp, task=task, extra=extra)
        return predictions


@register_body
def get_maskdino_head(cfg, input_shape, lang_encoder, extra):
    return OpenSeeDHead(cfg, input_shape, lang_encoder, extra)
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head, get_tr2_head
from pysot.models.neck import get_neck
import torchvision
from torch import Tensor
from pysot.models.head.transformer.criterion import Tr2Criterion
from torchvision.models._utils import IntermediateLayerGetter

class Backbone(nn.Module):
    def __init__(self, backbone_name:str="resnet50"):
        super().__init__()
        backbone = getattr(torchvision.models, backbone_name)(pretrained=True)
        return_layers = {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        for k, v in self.body.items():
            setattr(self, k, v)
        self.num_channels = 512 if backbone_name in ('resnet18', 'resnet34') else 2048
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.body(x)["layer4"]
        return out

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        if cfg.BACKBONE.CUSTOM_BACKBONE:
            self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        else:
            self.backbone = Backbone(cfg.BACKBONE.TYPE)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        if cfg.TRANSFORMER.TRANSFORMER:
            if cfg.ADJUST.ADJUST:
                cfg.TRANSFORMER.KWARGS["d_model"] = cfg.ADJUST.KWARGS.out_channels[-1]
            self.tr2_head = get_tr2_head(cfg.TRANSFORMER.TYPE,
                                        **cfg.TRANSFORMER.KWARGS)
            self.criterion = Tr2Criterion(cfg.TRAIN.CLS_WEIGHT, cfg.TRAIN.LOC_WEIGHT, cfg.TRAIN.IOU_WEIGHT)
        else:
            # build rpn head
            self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                        **cfg.RPN.KWARGS)

            # build mask head
            if cfg.MASK.MASK:
                self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                            **cfg.MASK.KWARGS)

                if cfg.REFINE.REFINE:
                    self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        

        if cfg.TRANSFORMER.TRANSFORMER:
            zf = self.backbone(template)
            xf = self.backbone(search)

            if len(cfg.ADJUST.KWARGS.out_channels) == 1:
                if isinstance(zf, list):
                    zf = zf[-1]
                    xf = xf[-1]
                if cfg.ADJUST.ADJUST:
                    zf = self.neck(zf)
                    xf = self.neck(xf)
            else:
                if isinstance(zf, list):
                    if cfg.ADJUST.ADJUST:
                        zf = self.neck(zf)
                        xf = self.neck(xf)
                    zf = zf[-1]
                    xf = xf[-1]
            x = self.tr2_head(zf, xf)
            outputs = self.criterion(x, (label_cls, label_loc))
            return outputs
        else:
            # get feature
            label_loc_weight = data['label_loc_weight'].cuda()
            zf = self.backbone(template)
            xf = self.backbone(search)
            if cfg.MASK.MASK:
                zf = zf[-1]
                self.xf_refine = xf[:-1]
                xf = xf[-1]
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)
                xf = self.neck(xf)
            cls, loc = self.rpn_head(zf, xf)

            # get loss
            cls = self.log_softmax(cls)
            cls_loss = select_cross_entropy_loss(cls, label_cls)
            loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

            outputs = {}
            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_loss
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss

            if cfg.MASK.MASK:
                # TODO
                mask, self.mask_corr_feature = self.mask_head(zf, xf)
                mask_loss = None
                outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
                outputs['mask_loss'] = mask_loss
            return outputs

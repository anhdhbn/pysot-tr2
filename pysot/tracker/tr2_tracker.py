from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
import cv2
import torch
from pysot.utils.bbox import center2corner
from pysot.models.head.transformer.box_ops import box_cxcywh_to_xyxy

class Tr2Tracker(SiameseTracker):
    def __init__(self, model):
        super(Tr2Tracker, self).__init__()
        self.model = model
        self.model.eval()

    def transform(self, img):
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :, :, :]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = img.cuda()
        return img

    def init_(self, img):
        """
        args:
            img(np.ndarray): BGR image
        """
        # get crop
        z_crop = self.transform(img)
        self.model.template(z_crop)
    
    def init(self, img, roi):
        (x, y, w, h) = roi
        z_crop = img[int(y):int(y+h), int(x):int(x+w)]
        self.init_(z_crop)

    def track(self, img, label_cls, label_loc):
        shape = img.shape[:2]
        x_crop = self.transform(img)
        cls, loc, loss = self.model.track(x_crop, label_cls, label_loc)

        cls = self._convert_score(cls)
        bbox = self._convert_bbox(loc.flatten(), shape)
        return (cls, bbox, loss)

    def _convert_score(self, score):
        return F.softmax(score, dim=1).data[:, 0].cpu().numpy()

    def _convert_bbox(self, delta, shape):
        # delta = box_cxcywh_to_xyxy(delta)
        # print("*"*30)
        # print(delta)

        h_s, w_s = shape
        cx, cy, w, h = delta.data.cpu().numpy()
        x1 = cx - w/2
        y1 = cy - h/2
        # print([x1, y1, w, h])

        x1 = int(x1 * w_s)
        y1 = int(y1 * h_s)
        w = int(w * w_s)
        h = int(h * h_s)
        out = [x1, y1, w, h]
        # print(out)
        # exit(0)
        return out
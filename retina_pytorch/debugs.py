from utils import box_iou
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageDraw2
from encoder import DataEncoder
from torch import Tensor as T
import torch
import imgaug as ia

def calc_area(box):
    w = np.abs(box[2] - box[0]) #xmin ymin xxmax ymax
    h = np.abs(box[3] - box[1])

    return w * h

im_size = (600,600)


encoder = DataEncoder()
canvas = np.ones(im_size+(3,))
image = Image.fromarray(np.uint8(canvas))
anchors,_ = encoder._get_anchor_boxes(T(im_size))

anchors = np.stack(sorted(list(anchors), key=lambda x: calc_area(x)))
small_anchors = anchors[:100, :]
medium_anchors = anchors[75000:75050, :]
big_anchors = anchors[-50::, :]

# small_anchors = [[548.      ,   36.      ,   57.017517,   28.508759]]


small_anchors_canvas = Image.fromarray(np.uint8(np.zeros(im_size + (3,))))

bbs = [ia.BoundingBox(*b) for b in big_anchors]
for b in bbs:
    small_anchors_canvas = b.draw_on_image(
        small_anchors_canvas, color=np.random.rand(3) * 255, thickness=2
    )

# plt.figure(figsize=(20, 20))
# plt.imshow(small_anchors_canvas)
#


some_box = T([[100,100,170,170]])

from encoder import change_box_order
input_size = T(im_size)
labels = T([0,0])
loc_targets, cls_targets = encoder.encode(some_box, labels, input_size)
print(sum(cls_targets>1))
print(len(anchors))


matched_anchors = anchors[np.array(cls_targets>0.1).astype(bool),:]

print(len(matched_anchors))
real_bbs_on_image = ia.BoundingBoxesOnImage(
    [ia.BoundingBox(*b) for b in some_box], shape=im_size)

matched_anchors_on_image = ia.BoundingBoxesOnImage(
    [ia.BoundingBox(*b) for b in matched_anchors], shape=im_size)


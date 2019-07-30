from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os

import sys
sys.path.append("coco-caption")
from pycocotools.coco import COCO
from pycocoevalcap.spice.spice import Spice

train_path = '/home/yangxu/project/self-critical.pytorch/data/coco_annotations/captions_train2014.json'
val_path = '/home/yangxu/project/self-critical.pytorch/data/coco_annotations/captions_val2014.json'

coco_train = COCO(train_path)
coco_val = COCO(val_path)

coco_use = coco_train

image_ids = coco_use.getImgIds()
gts = {}
res = {}
for img_id in image_ids:
	gts[img_id] = []
	data_temp = coco_use.imgToAnns[img_id]
	for dt in data_temp:
		gts[img_id].append(dt['caption'])
	res[img_id] = []
	res[img_id].append(gts[img_id][0])

scorer = Spice()
score, scores = scorer.compute_score(gts, res)

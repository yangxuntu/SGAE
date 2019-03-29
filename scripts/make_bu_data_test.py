from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='data/bu_data', help='downloaded feature directory')
parser.add_argument('--output_dir', default='data/cocobu_test', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infiles = ['test2014/test2014_resnet101_faster_rcnn_genome.tsv.0',
          'test2014/test2014_resnet101_faster_rcnn_genome.tsv.1',
          'test2014/test2014_resnet101_faster_rcnn_genome.tsv.2']

os.makedirs(args.output_dir+'_att')
os.makedirs(args.output_dir+'_fc')
os.makedirs(args.output_dir+'_box')
cocobu_size = {}

for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            item['image_w'] = int(item['image_w'])
            item['image_h'] = int(item['image_h'])
            for field in ['boxes','features']:
                try:
                    item[field] = np.frombuffer(base64.decodestring(item[field]),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))
                except:
                    print(item['image_id'])
            try:
                np.savez_compressed(os.path.join(args.output_dir+'_att', str(item['image_id'])), feat=item['features'])
                np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
                np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])
                cocobu_size_temp = {}
                cocobu_size_temp['image_w'] = item['image_w']
                cocobu_size_temp['image_h'] = item['image_h']
                cocobu_size[str(item['image_id'])] = cocobu_size_temp
            except:
                print(item['image_id'])
save_path = '/home/yangxu/project/self-critical.pytorch/data/cocobu_size_test.npz'
np.savez(save_path, roidb=cocobu_size)





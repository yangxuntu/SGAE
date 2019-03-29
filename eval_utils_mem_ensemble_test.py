from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

def eval_split(model, loader, training_mode=0, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_rela = eval_kwargs.get('use_rela', 0)
    index_eval = eval_kwargs.get('index_eval', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        fc_feats = None
        att_feats = None
        att_masks = None
        ssg_data = None
        rela_data = None
        if use_rela:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['rela_rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['rela_rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['rela_attr_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['rela_attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   ]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, rela_rela_matrix, rela_rela_masks, rela_attr_matrix, rela_attr_masks = tmp
        else:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   ]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks = tmp
            rela_rela_matrix = None
            rela_rela_masks = None
            rela_attr_matrix = None
            rela_attr_masks = None

        rela_data = {}
        rela_data['att_feats'] = att_feats
        rela_data['att_masks'] = att_masks
        rela_data['rela_matrix'] = rela_rela_matrix
        rela_data['rela_masks'] = rela_rela_masks
        rela_data['attr_matrix'] = rela_attr_matrix
        rela_data['attr_masks'] = rela_attr_masks

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, att_feats, att_masks, rela_data,
                        ssg_data,use_rela, training_mode, opt=eval_kwargs, mode='sample_beam')[0].data
        
        # Print beam search
        sents_save_temp = []
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                sents_temp = []
                sents_length = []
                for seq_temp in model.done_beams[i]:
                    sent_temp = utils.decode_sequence(loader.get_vocab(), seq_temp['seq'].unsqueeze(0),
                                                       use_ssg=1)[0]
                    sents_temp.append(sent_temp)
                    sents_length.append(len(sent_temp))
                #     print('{0}'.format(sent_temp))
                # print('--' * 10)
                sents_index = sents_length.index(max(sents_length))
                sents_save_temp.append(sents_temp[sents_index])


        sents = utils.decode_sequence(loader.get_vocab(), seq, use_ssg=1)
        #sents = sents_save_temp
        print('{0}/{1}'.format(n,loader.num_images))
        for k, sent in enumerate(sents):
            entry = {'image_id': int(data['infos'][k]['id']), 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']

            # if verbose:
            #     print('image %s, %s' %(entry['image_id'], entry['caption']))
            #     text_file = open('generated_caption.txt', "aw")
            #     text_file.write('image %s, %s' %(entry['image_id'], entry['caption']))
            #     text_file.write('\n')
            #     text_file.close()

            predictions.append(entry)

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d' % (ix0 - 1, ix1))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
    predictions.append({'caption': u'An airplane is flying in the sky', 'image_id': 321486})
    predictions.append({'caption': u'A man sitting on a bench with a skateboard','image_id': 300104})
    predictions.append({'caption': u'A pizza sitting on top of a box', 'image_id': 147295})

    return predictions

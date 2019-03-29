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

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, training_mode=0, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_rela = eval_kwargs.get('use_rela', 0)
    index_eval = eval_kwargs.get('index_eval', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            fc_feats = None
            att_feats = None
            att_masks = None
            ssg_data = None
            rela_data = None

            tmp = [data['fc_feats'], data['labels'], data['masks']]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, labels, masks = tmp

            tmp = [data['att_feats'], data['att_masks'], data['rela_rela_matrix'],
                   data['rela_rela_masks'], data['rela_attr_matrix'], data['rela_attr_masks']]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]

            att_feats, att_masks, rela_rela_matrix, rela_rela_masks, \
            rela_attr_matrix, rela_attr_masks = tmp

            rela_data = {}
            rela_data['att_feats'] = att_feats
            rela_data['att_masks'] = att_masks
            rela_data['rela_matrix'] = rela_rela_matrix
            rela_data['rela_masks'] = rela_rela_masks
            rela_data['attr_matrix'] = rela_attr_matrix
            rela_data['attr_masks'] = rela_attr_masks

            tmp = [data['ssg_rela_matrix'], data['ssg_rela_masks'], data['ssg_obj'], data['ssg_obj_masks'],
                   data['ssg_attr'], data['ssg_attr_masks']]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks = tmp
            ssg_data = {}
            ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
            ssg_data['ssg_rela_masks'] = ssg_rela_masks
            ssg_data['ssg_obj'] = ssg_obj
            ssg_data['ssg_obj_masks'] = ssg_obj_masks
            ssg_data['ssg_attr'] = ssg_attr
            ssg_data['ssg_attr_masks'] = ssg_attr_masks

            loss = 0
            # with torch.no_grad():
            #     loss = crit(model(fc_feats, att_feats, labels, att_masks,
            #                       rela_data, ssg_data,use_rela, training_mode), labels[:, 1:],masks[:, 1:]).item()

            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
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
                   data['ssg_rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   ]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, rela_rela_matrix, rela_rela_masks, rela_attr_matrix, rela_attr_masks, \
            ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks = tmp
        else:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   ]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, ssg_rela_matrix, ssg_rela_masks, \
            ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks = tmp
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
        ssg_data = {}
        ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
        ssg_data['ssg_rela_masks'] = ssg_rela_masks
        ssg_data['ssg_obj'] = ssg_obj
        ssg_data['ssg_obj_masks'] = ssg_obj_masks
        ssg_data['ssg_attr'] = ssg_attr
        ssg_data['ssg_attr_masks'] = ssg_attr_masks

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
                    print('{0}'.format(sent_temp))
                print('--' * 10)
                sents_index = sents_length.index(max(sents_length))
                sents_save_temp.append(sents_temp[sents_index])


        sents = utils.decode_sequence(loader.get_vocab(), seq, use_ssg=1)
        #sents = sents_save_temp
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']

            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                            data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
                    len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)



            if verbose:
                print('image %s, %s' %(entry['image_id'], entry['caption']))
                text_file = open('generated_caption.txt', "aw")
                text_file.write('image %s, %s' %(entry['image_id'], entry['caption']))
                text_file.write('\n')
                text_file.close()

            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats

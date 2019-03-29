# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel_mem import CaptionModel

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttModel_mem4_new(CaptionModel):
    def __init__(self, opt):
        super(AttModel_mem4_new, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.seq_per_img = opt.seq_per_img
        self.use_rela = getattr(opt, 'use_rela', 0)
        self.rela_dict_len = getattr(opt, 'rela_dict_size', 0)
        self.use_attr_info = getattr(opt, 'use_attr_info', 1)

        self.ssg_core = TopDownCore_mem(opt)
        self.rela_core = TopDownCore_rela(opt)
        self.img2sg = imge2sene_fc(opt)

        self.memory_index = getattr(opt, 'memory_index', 'c')
        self.memory_size = getattr(opt, 'memory_size', 5000)
        self.memory_cell_path = getattr(opt, 'memory_cell_path', '0')

        self.index_eval = getattr(opt, 'index_eval', 0)

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.drop_prob_lm))
        self.embed2vis = nn.Sequential(nn.Linear(self.input_encoding_size, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        self.rela_embed = nn.Sequential(nn.Embedding(self.rela_dict_len, self.input_encoding_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.drop_prob_lm))
        self.rela_sbj_rela_fc = nn.Sequential(nn.Linear(self.rnn_size*3, self.rnn_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))
        self.rela_obj_rela_fc = nn.Sequential(nn.Linear(self.rnn_size*3, self.rnn_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))
        self.rela_rela_fc = nn.Sequential(nn.Linear(self.rnn_size*3, self.rnn_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))
        self.rela_attr_fc = nn.Sequential(nn.Linear(self.rnn_size*2, self.rnn_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout(self.drop_prob_lm))

        self.ssg_sbj_rela_fc = nn.Sequential(nn.Linear(self.input_encoding_size * 3, self.rnn_size),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(self.drop_prob_lm))
        self.ssg_obj_rela_fc = nn.Sequential(nn.Linear(self.input_encoding_size*3, self.rnn_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.drop_prob_lm))
        self.ssg_obj_obj_fc = nn.Sequential(nn.Linear(self.input_encoding_size, self.rnn_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.drop_prob_lm))
        self.ssg_obj_attr_fc = nn.Sequential(nn.Linear(self.input_encoding_size*2, self.rnn_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.drop_prob_lm))
        self.ssg_rela_fc = nn.Sequential(nn.Linear(self.input_encoding_size*3, self.rnn_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.drop_prob_lm))
        self.ssg_attr_fc = nn.Sequential(nn.Linear(self.input_encoding_size*2, self.rnn_size),
                                nn.ReLU(inplace=True),
                                nn.Dropout(self.drop_prob_lm))


        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in
                              range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(
                *(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))

        self.rela_ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ssg_ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        if os.path.isfile(self.memory_cell_path):
            print('load memory_cell from {0}'.format(self.memory_cell_path))
            memory_init = np.load(self.memory_cell_path)['memory_cell'][()]
        else:
            print('create a new memory_cell')
            memory_init = np.random.rand(self.memory_size, self.rnn_size) / 100
        memory_init = np.float32(memory_init)
        self.memory_cell = torch.from_numpy(memory_init).cuda().requires_grad_()


        #self.rela_mem = Memory_cell2(opt)
        self.ssg_mem = Memory_cell2(opt)



    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        return fc_feats, att_feats


    def rela_graph_gfc(self, rela_data):
        """
        :param att_feats: roi features of each bounding box, [N_img*5, N_att_max, rnn_size]
        :param rela_feats: the embeddings of relationship, [N_img*5, N_rela_max, rnn_size]
        :param rela_matrix: relationship matrix, [N_img*5, N_rela_max, 3], N_img
                            is the batch size, N_rela_max is the maximum number
                            of relationship in rela_matrix.
        :param rela_masks: relationship masks, [N_img*5, N_rela_max].
                            For each row, the sum of that row is the total number
                            of realtionship.
        :param att_masks: attention masks, [N_img*5, N_att_max].
                            For each row, the sum of that row is the total number
                            of roi poolings.
        :param attr_matrix: attribute matrix,[N_img*5, N_attr_max, N_attr_each_max]
                            N_img is the batch size, N_attr_max is the maximum number
                            of attributes of one mini-batch, N_attr_each_max is the
                            maximum number of attributes of each objects in that mini-batch
        :param attr_masks: attribute masks, [N_img*5, N_attr_max, N_attr_each_max]
                            the sum of attr_masks[img_id*5,:,0] is the number of objects
                            which own attributes, the sum of attr_masks[img_id*5, obj_id, :]
                            is the number of attribute that object has
        :return: att_feats_new: new roi features
                 rela_feats_new: new relationship embeddings
                 attr_feats_new: new attribute features
        """
        att_feats = rela_data['att_feats']
        att_masks = rela_data['att_masks']
        rela_matrix = rela_data['rela_matrix']
        rela_feats = rela_data['rela_feats']
        rela_masks = rela_data['rela_masks']
        attr_matrix = rela_data['attr_matrix']
        attr_masks = rela_data['attr_masks']

        att_feats_size = att_feats.size()
        attr_masks_size = attr_masks.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att / seq_per_img
        att_feats_new = att_feats.clone()
        rela_feats_new = rela_feats.clone()
        attr_feats_new = torch.zeros([attr_masks_size[0], attr_masks_size[1], self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            N_box = torch.sum(att_masks[img_id * seq_per_img, :])
            N_rela = int(N_rela)
            N_box = int(N_box)
            box_num = np.ones([N_box,])
            rela_num = np.ones([N_rela,])
            for i in range(N_rela):
                sub_id = rela_matrix[img_id * seq_per_img, i, 0]
                sub_id = int(sub_id)
                box_num[sub_id] += 1.0
                obj_id = rela_matrix[img_id * seq_per_img, i, 1]
                obj_id = int(obj_id)
                box_num[obj_id] += 1.0
                rela_id = i
                rela_num[rela_id] += 1.0
                sub_feat_use = att_feats[img_id * seq_per_img, sub_id, :]
                obj_feat_use = att_feats[img_id * seq_per_img, obj_id, :]
                rela_feat_use = rela_feats[img_id * seq_per_img, rela_id, :]

                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, sub_id, :] += \
                    self.rela_sbj_rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))
                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, obj_id, :] += \
                    self.rela_obj_rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))
                rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, rela_id, :] += \
                    self.rela_rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))

            if self.use_attr_info == 1:
                N_obj_attr = torch.sum(attr_masks[img_id * seq_per_img, :, 0])
                N_obj_attr = int(N_obj_attr)
                for i in range(N_obj_attr):
                    attr_obj_id = int(attr_matrix[img_id * seq_per_img, i, 0])
                    obj_feat_use = att_feats[img_id * seq_per_img, int(attr_obj_id), :]
                    N_attr_each = torch.sum(attr_masks[img_id * seq_per_img, i, :])
                    for j in range(N_attr_each-1):
                        attr_index = attr_matrix[img_id * seq_per_img, i, j+1].cuda().long()
                        attr_feat_use = self.rela_embed(attr_index)
                        attr_feat_use = self.embed2vis(attr_feat_use)
                        attr_feats_new[img_id * seq_per_img:(img_id+1) * seq_per_img, i, :] += \
                            self.rela_attr_fc( torch.cat((attr_feat_use, obj_feat_use)) )
                    attr_feats_new[img_id * seq_per_img:(img_id+1) * seq_per_img, i, :] = \
                        attr_feats_new[img_id * seq_per_img:(img_id+1) * seq_per_img, i, :]/(float(N_attr_each)-1)


            for i in range(N_box):
                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i] = \
                    att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i]/box_num[i]
            for i in range(N_rela):
                rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i, :] = \
                    rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i, :]/rela_num[i]

        rela_data['att_feats'] = att_feats_new
        rela_data['rela_feats'] = rela_feats_new
        if self.use_attr_info == 1:
            rela_data['attr_feats'] = attr_feats_new
        return rela_data


    def prepare_rela_feats(self, rela_data):
        """
        Change relationship index (one-hot) to relationship features, or change relationship
        probability to relationship features.
        :param rela_matrix:
        :param rela_masks:
        :return: rela_features, [N_img*5, N_rela_max, rnn_size]
        """
        rela_matrix = rela_data['rela_matrix']
        rela_masks = rela_data['rela_masks']

        rela_feats_size = rela_matrix.size()
        N_att = rela_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att/seq_per_img
        rela_feats = torch.zeros([rela_feats_size[0], rela_feats_size[1], self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            N_rela = int(N_rela)
            if N_rela>0:
                rela_index = rela_matrix[img_id*seq_per_img,:N_rela,2].cuda().long()
                rela_feats_temp = self.rela_embed(rela_index)
                rela_feats_temp = self.embed2vis(rela_feats_temp)
                rela_feats[img_id*seq_per_img:(img_id+1)*seq_per_img,:N_rela,:] = rela_feats_temp
        rela_data['rela_feats'] = rela_feats
        return rela_data

    def merge_rela_att(self, rela_data):
        """
        merge attention features (roi features) and relationship features together
        :param att_feats: [N_att, N_att_max, rnn_size]
        :param att_masks: [N_att, N_att_max]
        :param rela_feats: [N_att, N_rela_max, rnn_size]
        :param rela_masks: [N_att, N_rela_max]
        :return: att_feats_new: [N_att, N_att_new_max, rnn_size]
                 att_masks_new: [N_att, N_att_new_max]
        """
        att_feats = rela_data['att_feats']
        att_masks = rela_data['att_masks']
        rela_feats = rela_data['rela_feats']
        rela_masks = rela_data['rela_masks']
        if self.use_attr_info == 1:
            attr_feats = rela_data['attr_feats']
            attr_masks = rela_data['attr_masks']

        att_feats_size = att_feats.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att/seq_per_img
        N_att_new_max = -1
        for img_id in range(int(N_img)):
            if self.use_attr_info != 0:
                N_att_new_max = \
                max(N_att_new_max,torch.sum(rela_masks[img_id * seq_per_img, :]) +
                    torch.sum(att_masks[img_id * seq_per_img, :]) + torch.sum(attr_masks[img_id * seq_per_img,:,0]))
            else:
                N_att_new_max = \
                    max(N_att_new_max, torch.sum(rela_masks[img_id * seq_per_img, :]) +
                        torch.sum(att_masks[img_id * seq_per_img, :]))
        att_masks_new = torch.zeros([N_att, int(N_att_new_max)]).cuda()
        att_feats_new = torch.zeros([N_att, int(N_att_new_max), self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = int(torch.sum(rela_masks[img_id * seq_per_img, :]))
            N_box = int(torch.sum(att_masks[img_id * seq_per_img, :]))
            if self.use_attr_info == 1:
                N_attr = int(torch.sum(attr_masks[img_id * seq_per_img,:,0]))
            else:
                N_attr = 0
            att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box, :] = \
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box, :]
            if N_rela > 0:
                att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box:N_box + N_rela, :] = \
                    rela_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_rela, :]
            if N_attr > 0:
                att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box + N_rela: N_box + N_rela + N_attr, :] = \
                    attr_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_attr, :]
            att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box] = 1
            if N_rela > 0:
                att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box:N_box + N_rela] = 1
            if N_attr > 0:
                att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box + N_rela:N_box + N_rela + N_attr] = 1

        rela_data['att_feats_new'] = att_feats_new
        rela_data['att_masks_new'] = att_masks_new
        return rela_data

    def ssg_graph_gfc(self, ssg_data):
        """
        use sentence scene graph's graph network to embed feats,
        :param ssg_data: one dict which contains the following data:
               ssg_data['ssg_rela_matrix']: relationship matrix for ssg data,
                    [N_att, N_rela_max, 3] array
               ssg_data['ssg_rela_masks']: relationship masks for ssg data,
                    [N_att, N_rela_max]
               ssg_data['ssg_obj']: obj index for ssg data, [N_att, N_obj_max]
               ssg_data['ssg_obj_masks']: obj masks, [N_att, N_obj_max]
               ssg_data['ssg_attr']: attribute indexes, [N_att, N_obj_max, N_attr_max]
               ssg_data['ssg_attr_masks']: attribute masks, [N_att, N_obj_max, N_attr_max]
        :return: ssg_data_new one dict which contains the following data:
                 ssg_data_new['ssg_rela_feats']: relationship embeddings, [N_att, N_rela_max, rnn_size]
                 ssg_data_new['ssg_rela_masks']: equal to ssg_data['ssg_rela_masks']
                 ssg_data_new['ssg_obj_feats']: obj embeddings, [N_att, N_obj_max, rnn_size]
                 ssg_data_new['ssg_obj_masks']: equal to ssg_data['ssg_obj_masks']
                 ssg_data_new['ssg_attr_feats']: attributes embeddings, [N_att, N_attr_max, rnn_size]
                 ssg_data_new['ssg_attr_masks']: equal to ssg_data['ssg_attr_masks']
        """
        ssg_data_new = {}
        ssg_data_new['ssg_rela_masks'] = ssg_data['ssg_rela_masks']
        ssg_data_new['ssg_obj_masks'] = ssg_data['ssg_obj_masks']
        ssg_data_new['ssg_attr_masks'] = ssg_data['ssg_attr_masks']

        ssg_obj = ssg_data['ssg_obj']
        ssg_obj_masks = ssg_data['ssg_obj_masks']
        ssg_attr = ssg_data['ssg_attr']
        ssg_attr_masks = ssg_data['ssg_attr_masks']
        ssg_rela_matrix = ssg_data['ssg_rela_matrix']
        ssg_rela_masks = ssg_data['ssg_rela_masks']

        ssg_obj_feats = torch.zeros([ssg_obj.size()[0], ssg_obj.size()[1], self.rnn_size]).cuda()
        ssg_rela_feats = torch.zeros([ssg_rela_matrix.size()[0], ssg_rela_matrix.size()[1], self.rnn_size]).cuda()
        ssg_attr_feats = torch.zeros([ssg_attr.size()[0], ssg_attr.size()[1], self.rnn_size]).cuda()
        ssg_attr_masks_new = torch.zeros(ssg_obj.size()).cuda()

        ssg_obj_size = ssg_obj.size()
        N_att = ssg_obj_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = int(N_att/seq_per_img)

        for img_id in range(N_img):
            N_obj = int(torch.sum(ssg_obj_masks[img_id*seq_per_img,:]))
            if N_obj == 0:
                continue
            obj_feats_ori = self.embed(ssg_obj[img_id*seq_per_img,:N_obj].cuda().long())
            obj_feats_temp = self.ssg_obj_obj_fc(obj_feats_ori)
            obj_num = np.ones([N_obj,])

            N_rela = int(torch.sum(ssg_rela_masks[img_id*seq_per_img,:]))
            rela_feats_temp = torch.zeros([N_rela, self.rnn_size])
            for rela_id in range(N_rela):
                sbj_id = int(ssg_rela_matrix[img_id * seq_per_img, rela_id, 0])
                obj_id = int(ssg_rela_matrix[img_id * seq_per_img, rela_id, 1])
                rela_index = ssg_rela_matrix[img_id * seq_per_img, rela_id, 2]
                sbj_feat = obj_feats_ori[sbj_id]
                obj_feat = obj_feats_ori[obj_id]
                rela_feat = self.embed(rela_index.cuda().long())
                obj_feats_temp[sbj_id] = obj_feats_temp[sbj_id] + self.ssg_sbj_rela_fc(torch.cat((sbj_feat, obj_feat, rela_feat)))
                obj_num[sbj_id] = obj_num[sbj_id] + 1.0
                obj_feats_temp[obj_id] = obj_feats_temp[obj_id] + self.ssg_obj_rela_fc(torch.cat((sbj_feat, obj_feat, rela_feat)))
                obj_num[obj_id] = obj_num[obj_id] + 1.0
                rela_feats_temp[rela_id] = self.ssg_rela_fc(torch.cat((sbj_feat, obj_feat, rela_feat)))
            for obj_id in range(N_obj):
                obj_feats_temp[obj_id] = obj_feats_temp[obj_id]/obj_num[obj_id]

            attr_feats_temp = torch.zeros([N_obj, self.rnn_size]).cuda()
            obj_attr_ids = 0
            for obj_id in range(N_obj):
                N_attr = int(torch.sum(ssg_attr_masks[img_id*seq_per_img, obj_id,:]))
                if N_attr != 0:
                    attr_feat_ori = self.embed(ssg_attr[img_id * seq_per_img, obj_id, :N_attr].cuda().long())
                    for attr_id in range(N_attr):
                        attr_feats_temp[obj_attr_ids] = attr_feats_temp[obj_attr_ids] +\
                                                        self.ssg_attr_fc(torch.cat((obj_feats_ori[obj_id], attr_feat_ori[attr_id])))
                    attr_feats_temp[obj_attr_ids] = attr_feats_temp[obj_attr_ids]/(N_attr + 0.0)
                    obj_attr_ids += 1
            N_obj_attr = obj_attr_ids
            ssg_attr_masks_new[img_id*seq_per_img:(img_id+1)*seq_per_img, :N_obj_attr] = 1

            ssg_obj_feats[img_id * seq_per_img: (img_id+1) * seq_per_img, :N_obj, :] = obj_feats_temp
            if N_rela != 0:
               ssg_rela_feats[img_id * seq_per_img: (img_id+1) * seq_per_img, :N_rela, :] = rela_feats_temp
            if N_obj_attr != 0:
                ssg_attr_feats[img_id * seq_per_img: (img_id+1) * seq_per_img, :N_obj_attr, :] = attr_feats_temp[:N_obj_attr]


        ssg_data_new['ssg_obj_feats'] = ssg_obj_feats
        ssg_data_new['ssg_rela_feats'] = ssg_rela_feats
        ssg_data_new['ssg_attr_feats'] = ssg_attr_feats
        ssg_data_new['ssg_attr_masks'] = ssg_attr_masks_new
        return ssg_data_new

    def merge_ssg_att(self, ssg_data_new):
        """
        merge ssg_obj_feats, ssg_rela_feats, ssg_attr_feats together
        :param ssg_data_new:
        :return: att_feats: [N_att, N_att_max, rnn_size]
                 att_masks: [N_att, N_att_max]
        """
        ssg_obj_feats = ssg_data_new['ssg_obj_feats']
        ssg_rela_feats = ssg_data_new['ssg_rela_feats']
        ssg_attr_feats = ssg_data_new['ssg_attr_feats']
        ssg_rela_masks = ssg_data_new['ssg_rela_masks']
        ssg_obj_masks = ssg_data_new['ssg_obj_masks']
        ssg_attr_masks = ssg_data_new['ssg_attr_masks']

        ssg_obj_size = ssg_obj_feats.size()
        N_att = ssg_obj_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = int(N_att / seq_per_img)

        N_att_max = -1
        for img_id in range(N_img):
            N_rela = int(torch.sum(ssg_rela_masks[img_id*seq_per_img,:]))
            N_obj = int(torch.sum(ssg_obj_masks[img_id*seq_per_img,:]))
            N_attr = int(torch.sum(ssg_attr_masks[img_id*seq_per_img,:]))
            N_att_max = max(N_att_max, N_rela + N_obj + N_attr)

        att_feats = torch.zeros([N_att, N_att_max, self.rnn_size]).cuda()
        att_masks = torch.zeros([N_att, N_att_max]).cuda()

        for img_id in range(N_img):
            N_rela = int(torch.sum(ssg_rela_masks[img_id * seq_per_img, :]))
            N_obj = int(torch.sum(ssg_obj_masks[img_id * seq_per_img, :]))
            N_attr = int(torch.sum(ssg_attr_masks[img_id * seq_per_img, :]))
            if N_obj != 0:
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, :N_obj, :] = \
                    ssg_obj_feats[img_id * seq_per_img, :N_obj, :]
            if N_rela != 0:
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_obj:N_obj+N_rela, :] = \
                    ssg_rela_feats[img_id * seq_per_img, :N_rela, :]

            if N_attr != 0:
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_obj+N_rela:N_obj+N_attr+N_rela, :] = \
                    ssg_attr_feats[img_id * seq_per_img, :N_attr, :]
            att_masks[img_id * seq_per_img:(img_id + 1) * seq_per_img, :N_obj+N_rela+N_attr] = 1

        ssg_data_new['att_feats_new'] = att_feats
        ssg_data_new['att_masks_new'] = att_masks
        return ssg_data_new



    def _forward(self, fc_feats, att_feats, seq, att_masks=None, rela_data=None, ssg_data=None, use_rela =1, training_mode = 0):
        """

        :param fc_feats:
        :param att_feats:
        :param seq:
        :param att_masks:
        :param rela_data:
        :param ssg_data:
        :param training_mode: when this is 0, using sentence sg and do not use memory,
               when this is 1, using sentence sg and write data into memory,
               when this is 2, using image sg and read data from memory
        :return:
        """
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        self.index_eval = 0
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        fc_feats, att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        if training_mode == 0:
            ssg_data = self.ssg_graph_gfc(ssg_data)
            ssg_data = self.merge_ssg_att(ssg_data)

            att_feats_mem = ssg_data['att_feats_new']
            att_masks = ssg_data['att_masks_new']
            p_att_feats_mem = self.ssg_ctx2att(att_feats_mem)
            att_feats_rela = None
            p_att_feats_rela = None

        if training_mode == 1:
            ssg_data = self.ssg_graph_gfc(ssg_data)
            ssg_data = self.merge_ssg_att(ssg_data)

            att_feats_mem = ssg_data['att_feats_new']
            att_feats_mem = self.ssg_mem(att_feats_mem, self.memory_cell)
            att_masks = ssg_data['att_masks_new']
            p_att_feats_mem = self.ssg_ctx2att(att_feats_mem)
            att_feats_rela = None
            p_att_feats_rela = None

        if training_mode == 2:
            if use_rela == 1:
                rela_data['att_feats'] = att_feats
                rela_data['att_masks'] = att_masks
                rela_data = self.prepare_rela_feats(rela_data)
                rela_data = self.rela_graph_gfc(rela_data)
                rela_data = self.merge_rela_att(rela_data)
            else:
                rela_data['att_feats_new'] = att_feats
                rela_data['att_masks_new'] = att_masks

            att_feats_rela = rela_data['att_feats_new']
            att_feats_mem = self.img2sg(att_feats_rela)
            att_feats_mem = self.ssg_mem(att_feats_mem, self.memory_cell)
            #att_feats = self.rela_mem(att_feats, self.memory_cell)
            att_masks = rela_data['att_masks_new']
            p_att_feats_rela = self.rela_ctx2att(att_feats_rela)
            p_att_feats_mem = self.rela_ctx2att(att_feats_mem)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob

                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].detach())
                    it.index_copy_(0, sample_ind,
                                        torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()

            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(
                it, fc_feats, att_feats_mem, p_att_feats_mem, att_feats_rela,
                p_att_feats_rela, att_masks, training_mode, state)

            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats_mem, p_att_feats_mem, att_feats_rela,
                           p_att_feats_rela, att_masks, training_mode, state):
        # 'it' contains a word index

        xt = self.embed(it)

        if training_mode == 0 or training_mode == 1:
            output, state = self.ssg_core(xt, fc_feats, att_feats_mem, p_att_feats_mem, state, att_masks)
        elif training_mode == 2:
            output, state = self.rela_core(xt, fc_feats, att_feats_mem, att_feats_rela,
                                          p_att_feats_mem,  p_att_feats_rela, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state


    def _sample_beam(self, fc_feats, att_feats, att_masks=None, rela_data = None, ssg_data=None, use_rela=1, training_mode=0, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        fc_feats, att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        if training_mode == 0:
            ssg_data = self.ssg_graph_gfc(ssg_data)
            ssg_data = self.merge_ssg_att(ssg_data)

            att_feats_mem = ssg_data['att_feats_new']
            att_masks = ssg_data['att_masks_new']
            p_att_feats_mem = self.ssg_ctx2att(att_feats_mem)
            att_feats_rela = None
            p_att_feats_rela = None

        if training_mode == 1:
            ssg_data = self.ssg_graph_gfc(ssg_data)
            ssg_data = self.merge_ssg_att(ssg_data)

            att_feats_mem = ssg_data['att_feats_new']
            att_feats_mem = self.ssg_mem(att_feats_mem, self.memory_cell)
            att_masks = ssg_data['att_masks_new']
            p_att_feats_mem = self.ssg_ctx2att(att_feats_mem)
            att_feats_rela = None
            p_att_feats_rela = None

        if training_mode == 2:
            if use_rela == 1:
                rela_data['att_feats'] = att_feats
                rela_data['att_masks'] = att_masks
                rela_data = self.prepare_rela_feats(rela_data)
                rela_data = self.rela_graph_gfc(rela_data)
                rela_data = self.merge_rela_att(rela_data)
            else:
                rela_data['att_feats_new'] = att_feats
                rela_data['att_masks_new'] = att_masks

            att_feats_rela = rela_data['att_feats_new']
            att_feats_mem = self.img2sg(att_feats_rela)
            att_feats_mem = self.ssg_mem(att_feats_mem, self.memory_cell)
            #att_feats = self.rela_mem(att_feats, self.memory_cell)
            att_masks = rela_data['att_masks_new']
            p_att_feats_rela = self.rela_ctx2att(att_feats_rela)
            p_att_feats_mem = self.rela_ctx2att(att_feats_mem)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)

            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats_mem = att_feats_mem[k:k + 1].expand(*((beam_size,) + att_feats_mem.size()[1:])).contiguous()
            tmp_p_att_feats_mem = p_att_feats_mem[k:k + 1].expand(
                *((beam_size,) + p_att_feats_mem.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k + 1].expand(
                *((beam_size,) + att_masks.size()[1:])).contiguous() if att_masks is not None else None

            if training_mode == 2:
                tmp_att_feats_rela = att_feats_rela[k:k + 1].expand(
                    *((beam_size,) + att_feats_rela.size()[1:])).contiguous()
                tmp_p_att_feats_rela = p_att_feats_rela[k:k + 1].expand(
                    *((beam_size,) + p_att_feats_rela.size()[1:])).contiguous()
            else:
                tmp_att_feats_rela = None
                tmp_p_att_feats_rela = None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(
                    it, tmp_fc_feats, tmp_att_feats_mem, tmp_p_att_feats_mem, tmp_att_feats_rela,
                    tmp_p_att_feats_rela, tmp_att_masks, training_mode, state)

            self.done_beams[k] = self.beam_search(
                state, logprobs, training_mode, tmp_fc_feats, tmp_att_feats_mem, tmp_p_att_feats_mem,
                tmp_att_feats_rela, tmp_p_att_feats_rela, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, rela_data=None, ssg_data=None, use_rela = 1, training_mode=0, opt={}):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, rela_data, ssg_data, use_rela, training_mode, opt)

        self.index_eval = 1
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        fc_feats, att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        if training_mode == 0:
            ssg_data = self.ssg_graph_gfc(ssg_data)
            ssg_data = self.merge_ssg_att(ssg_data)

            att_feats_mem = ssg_data['att_feats_new']
            att_masks = ssg_data['att_masks_new']
            p_att_feats_mem = self.ssg_ctx2att(att_feats_mem)
            att_feats_rela = None
            p_att_feats_rela = None

        if training_mode == 1:
            ssg_data = self.ssg_graph_gfc(ssg_data)
            ssg_data = self.merge_ssg_att(ssg_data)

            att_feats_mem = ssg_data['att_feats_new']
            att_feats_mem = self.ssg_mem(att_feats_mem, self.memory_cell)
            att_masks = ssg_data['att_masks_new']
            p_att_feats_mem = self.ssg_ctx2att(att_feats_mem)
            att_feats_rela = None
            p_att_feats_rela = None

        if training_mode == 2:
            if use_rela == 1:
                rela_data['att_feats'] = att_feats
                rela_data['att_masks'] = att_masks
                rela_data = self.prepare_rela_feats(rela_data)
                rela_data = self.rela_graph_gfc(rela_data)
                rela_data = self.merge_rela_att(rela_data)
            else:
                rela_data['att_feats_new'] = att_feats
                rela_data['att_masks_new'] = att_masks

            att_feats_rela = rela_data['att_feats_new']
            att_feats_mem = self.img2sg(att_feats_rela)
            att_feats_mem = self.ssg_mem(att_feats_mem, self.memory_cell)
            #att_feats = self.rela_mem(att_feats, self.memory_cell)
            att_masks = rela_data['att_masks_new']
            p_att_feats_rela = self.rela_ctx2att(att_feats_rela)
            p_att_feats_mem = self.rela_ctx2att(att_feats_mem)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)

            logprobs, state = self.get_logprobs_state(
                it, fc_feats, att_feats_mem, p_att_feats_mem, att_feats_rela,
                p_att_feats_rela, att_masks, training_mode, state)

            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

        return seq, seqLogprobs

class TopDownCore_mem(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore_mem, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        #self.att_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        #self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        #att_lstm_input = torch.cat([prev_h, xt], 1)
        #att_lstm_input = torch.cat([fc_feats, xt], 1)

        # state[0][0] means the hidden state c in first lstm
        # state[1][0] means the cell h in first lstm
        # state[0] means hidden state and state[1] means cell, state[0][i] means
        # the i-th layer's hidden state
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

class TopDownCore_rela(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore_rela, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v, \hat_rela
        self.attention_mem = Attention(opt)
        self.attention_rela = Attention(opt)

    def forward(self, xt, fc_feats, att_feats_mem, att_feats_rela, p_att_feats_mem, p_att_feats_rela, state, att_masks):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        # state[0][0] means the hidden state c in first lstm
        # state[1][0] means the cell h in first lstm
        # state[0] means hidden state and state[1] means cell, state[0][i] means
        # the i-th layer's hidden state
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att_mem = self.attention_mem(h_att, att_feats_mem, p_att_feats_mem, att_masks)
        att_rela = self.attention_rela(h_att, att_feats_rela, p_att_feats_rela, att_masks)

        lang_lstm_input = torch.cat([att_mem, att_rela, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


class Memory_cell(nn.Module):
    def __init__(self, opt):
        """
        a_i = W^T*tanh(W_h*h + W_M*m_i)
        a_i: 1*1
        W: V*1
        W_h: V*R
        h: R*1
        W_M: V*R
        m_i: R*1
        M=[m_1,m_2,...,m_K]^T: K*R
        att = softmax(a): K*1
        h_out = M*att

        :param opt:
        """
        super(Memory_cell, self).__init__()
        self.R = opt.rnn_size
        self.V = opt.att_hid_size

        self.W = nn.Linear(self.V,1)
        self.W_h = nn.Linear(self.R, self.V)
        self.W_M = nn.Linear(self.R, self.V)

    def forward(self, h, M):
        M_size = M.size() #K*R
        h_size = h.size() #N*R

        att_h = self.W_h(h) #h:N*R att_h:N*V
        att_h = att_h.unsqueeze(1).expand([h_size[0],M_size[0],self.V]) #N*K*V

        M_expand = M.unsqueeze(0).expand([h_size[0],M_size[0],self.R]) #N*K*R
        att_M = self.W_M(M_expand) #N*K*V

        dot = att_h + att_M #N*K*V
        dot = F.tanh(att_M) #N*K*V
        dot = dot.view(-1, self.V)   #(N*K)*V
        dot = self.W(dot)   #N*K*1
        dot = dot.view(-1, M_size[0]) #N*K

        att = F.softmax(dot, dim=1) #N*K
        att_max = torch.max(att,dim=1)
        max_index = torch.argmax(att,dim=1)
        att_res = torch.bmm(att.unsqueeze(1), M_expand) # N*1*K, N*K*R->N*1*R
        att_res = att_res.squeeze(1) #N*R
        return att_res

class Memory_cell2(nn.Module):
    def __init__(self, opt):
        """
        a_i = h^T*m_i
        a_i: 1*1

        h: R*1
        m_i: R*1
        M=[m_1,m_2,...,m_K]^T: K*R
        att = softmax(a): K*1
        h_out = M*att: N*R

        :param opt:
        """
        super(Memory_cell2, self).__init__()
        self.R = opt.rnn_size
        self.V = opt.att_hid_size

        self.W = nn.Linear(self.V, 1)

    def forward(self, h, M):
        M_size = M.size()  # K*R
        h_size = h.size()  # N*T*R
        h = h.view(-1, h_size[2]) # (N*T)*R
        att = torch.mm(h, torch.t(M)) #(N*T)*K
        att = F.softmax(att, dim=1) #(N*T)*K
        #att_sum = torch.sum(att, dim=1)
        att_max = torch.max(att,dim=1)
        max_index = torch.argmax(att,dim=1)
        att_res = torch.mm(att, M)  #(N*T)*K * K*R->(N*T)*R
        att_res = att_res.view([h_size[0], h_size[1], h_size[2]])
        return att_res


class LSTM_mem4_new(AttModel_mem4_new):
    def __init__(self, opt):
        super(LSTM_mem4_new, self).__init__(opt)
        self.num_layers = 2

class imge2sene_fc(nn.Module):
    def __init__(self, opt):
        super(imge2sene_fc, self).__init__()
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm

        self.fc1 = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                 nn.LeakyReLU(0.1,inplace=True),
                                 nn.Dropout(self.drop_prob_lm))
        self.fc2 = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Dropout(self.drop_prob_lm))
        self.fc3 = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(y1)
        y3 = self.fc3(y2)
        return y3


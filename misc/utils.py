from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq, use_ssg = False):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                if use_ssg:
                    txt = txt + ix_to_word[ix.item()]
                else:
                    txt = txt + ix_to_word[str(ix.item())]
                # if  type(tags) != type(None):
                #     txt = txt + '('+str(np.int32(tags[i,j].item()))+')'
            else:
                break
        out.append(txt)
    return out

def decode_sequence_tags(ix_to_word, seq, use_ssg = False, tags=None):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                if use_ssg:
                    txt = txt + ix_to_word[ix.item()]
                else:
                    txt = txt + ix_to_word[str(ix.item())]
                if  type(tags) != type(None):
                    txt = txt + '('+str(np.int32(tags[i,j].item()))+')'
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

#accuracy for cross-entropy loss
class CE_ac(nn.Module):
    def __init__(self):
        super(CE_ac, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        pred_word = torch.argmax(input, dim=2)
        output = (pred_word == target).float()*mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class SgCriterion_weak(nn.Module):
    def __init__(self):
        super(SgCriterion_weak, self).__init__()

    def forward(self, input, target):
        """
        :param input: batch_size* class_size
        :param target: batch_size* class_size
        :return: loss
        """
        output = torch.log(target*(input-0.5)+0.5)
        output = -torch.sum(output)/input.size(0)
        return output

class SgCriterion(nn.Module):
    def __init__(self):
        super(SgCriterion, self).__init__()

    def forward(self, input, target, mask, train_mode):
        """
        :param input: batch_size* max_att* class_size
        :param target: batch_size* max_att
               mask: batch_size*max_att
        :return: loss
        """
        if train_mode == 'attr':
            input_inv = 1-input
            target_inv = 1-target
            output = -(torch.log(input)*target.float() + torch.log(input_inv)*target_inv.float())
            output = torch.sum(output,dim=2)
            output = torch.sum(output*mask)/torch.sum(mask)
            #target: batch_size * max_att *class_size
            #input: batch_size* max_att* class_size
        else:
            input_temp = -input.gather(2, target.long().unsqueeze(2)).squeeze(2)
            output = torch.sum(input_temp*mask)/torch.sum(mask)
        return output

def SgMae(input, target, mask, top, train_mode):
    """
    :param input: batch_size* max_att* class_size
    :param target: batch_size* max_att
           mask: batch_size*max_att
    :return: loss
    """
    sort_value, sort_index = torch.sort(input, dim=2, descending=True)
    sort_index = sort_index.cpu().numpy()
    top_objs = sort_index[:, :, :top]
    data_size = input.size()
    output = 0
    num_label = 0
    if train_mode == 'attr':
        for batch_id in range(data_size[0]):
            box_len = np.sum(mask[batch_id])
            for j in range(box_len):
                attr_label = np.where(target[batch_id,j] == 1)[0]
                attr_len = len(attr_label)
                for i in range(attr_len):
                    index = 0
                    num_label += attr_len
                    for k in range(top):
                        if attr_label[i] == top_objs[batch_id,j,k]:
                            index = 1
                    output += index
    else:
        for batch_id in range(data_size[0]):
            box_len = np.sum(mask[batch_id])
            num_label += box_len
            for j in range(box_len):
                for k in range(top):
                    if top_objs[batch_id,j,k] == target[batch_id,j]:
                        output += 1

    return output, num_label, top_objs

def SgPred(input, top, train_mode):
    """
    :param input: batch_size* max_att* class_size
    :param target: batch_size* max_att
           mask: batch_size*max_att
    :return: loss
    """
    sort_value, sort_index = torch.sort(input, dim=2, descending=True)
    sort_index = sort_index.cpu().numpy()
    top_objs = sort_index[:, :, :top]
    return top_objs

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if type(param.grad) != type(None):
                param.grad.data.clamp_(-grad_clip, grad_clip)

def compute_diff(diff, mask):
    output = torch.sum(diff*mask)/torch.sum(mask)
    return output

def compute_dis_diff(dis1, dis2, mask):
    loss_temp = torch.sum((dis1 - dis2).pow(2), 2)/dis1.size(2)
    loss = torch.sum(loss_temp*mask)/torch.sum(mask)
    return loss

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
    
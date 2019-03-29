from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
from models.ass_fun import *

import torch
import torch.utils.data as data

import multiprocessing

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_rela_dict_size(self):
        return self.rela_dict_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        
        # feature related options
        self.use_att = getattr(opt, 'use_att', True)
        self.use_rela = getattr(opt, 'use_rela', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        if self.opt.sg_dict_path == 'data/spice_sg_dict2.npz':
            sg_dict_info = np.load(self.opt.sg_dict_path)['spice_dict'][()]
            self.ix_to_word = sg_dict_info['ix_to_word']
        else:
            sg_dict_info = np.load(self.opt.sg_dict_path)['sg_dict'][()]
            self.ix_to_word = sg_dict_info['sg_ix_to_word']

        self.vocab_size = len(self.ix_to_word)
        if self.use_rela:
            self.input_rela_dir = self.opt.input_rela_dir

        if self.use_rela:
            self.rela_dict_dir = self.opt.rela_dict_dir
            rela_dict_info = np.load(self.rela_dict_dir)
            rela_dict = rela_dict_info[()]['rela_dict']
            self.rela_dict_size = len(rela_dict)
            print('rela dict size is {0}'.format(self.rela_dict_size))

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.input_box_dir = self.opt.input_box_dir

        self.seq_length = 16
        print('max sequence length in data is', self.seq_length)

        self.img_files = os.listdir(self.input_att_dir)
        self.img_ids = []

        self.num_images = len(self.img_files)

        print('read %d image features' %(self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'test': []}
        for ix in range(self.num_images):
            self.split_ix['test'].append(ix)
            self.img_ids.append(self.img_files[ix][:-4])

        print('assigned %d images to split test' %len(self.split_ix['test']))

        self.iterators = {'test': 0}
        
        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = [] # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = [] # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')


        rela_rela_batch = []
        rela_attr_batch = []

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, tmp_rela,\
                ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            rela_rela_batch.append(tmp_rela['rela_rela_matrix'])
            rela_attr_batch.append(tmp_rela['rela_attr_matrix'])
            if tmp_wrapped:
                wrapped = True

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.img_ids[ix]
            infos.append(info_dict)

        fc_batch, att_batch, infos = \
            zip(*sorted(zip(fc_batch, att_batch, infos), key=lambda x: 0, reverse=True))
        data = {}
        data['fc_feats'] = np.stack(reduce(lambda x,y:x+y, [[_]*seq_per_img for _ in fc_batch]))
        max_att_len = max([_.shape[0] for _ in att_batch])

        # merge att_feats
        data['att_feats'] = np.zeros([len(att_batch) * seq_per_img, max_att_len, att_batch[0].shape[1]],
                                     dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        if self.use_rela:
            max_rela_len = max([_.shape[0] for _ in rela_rela_batch])
            data['rela_rela_matrix'] = np.zeros([len(att_batch)*seq_per_img, max_rela_len, 3])
            for i in range(len(rela_rela_batch)):
                data['rela_rela_matrix'][i*seq_per_img:(i+1)*seq_per_img,0:len(rela_rela_batch[i]),:] = rela_rela_batch[i]
            data['rela_rela_masks'] = np.zeros(data['rela_rela_matrix'].shape[:2], dtype='float32')
            for i in range(len(rela_rela_batch)):
                data['rela_rela_masks'][i*seq_per_img:(i+1)*seq_per_img,:rela_rela_batch[i].shape[0]] = 1

            max_attr_obj_len = max(_.shape[0] for _ in rela_attr_batch)
            max_attr_each_len = max(_.shape[1] for _ in rela_attr_batch)
            data['rela_attr_masks'] = np.zeros([len(att_batch)*seq_per_img, max_attr_obj_len, max_attr_each_len], dtype='float32')
            data['rela_attr_matrix'] = np.zeros([len(att_batch)*seq_per_img, max_attr_obj_len, max_attr_each_len], dtype='float32')
            for i in range(len(rela_attr_batch)):
                data['rela_attr_matrix'][i*seq_per_img:(i+1)*seq_per_img,0:len(rela_attr_batch[i]),:rela_attr_batch[i].shape[1]] = \
                    rela_attr_batch[i]
            for i in range(len(rela_attr_batch)):
                attr_obj_len = rela_attr_batch[i].shape[0]
                for j in range(attr_obj_len):
                    attr_each_len = np.sum(rela_attr_batch[i][j,:]>=0)
                    data['rela_attr_masks'][i*seq_per_img:(i+1)*seq_per_img,j,:attr_each_len] = 1
        else:
            data['rela_rela_matrix'] = None
            data['rela_rela_masks'] = None
            data['rela_attr_matrix'] = None
            data['rela_attr_masks'] = None

        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index #self.split_ix[index]
        rela_data = {}
        rela_data['rela_rela_matrix'] = []
        rela_data['rela_attr_matrix'] = []

        if self.use_att:
            att_feat = np.load(os.path.join(self.input_att_dir, str(self.img_ids[ix]) + '.npz'))['feat']
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)

            if self.use_rela:
                path_temp = os.path.join(self.input_rela_dir, str(self.img_ids[ix]) + '.npy')
                if os.path.isfile(path_temp):
                    rela_info = np.load(os.path.join(path_temp))
                    rela_data['rela_rela_matrix'] = rela_info[()]['rela_matrix']
                    rela_data['rela_attr_matrix'] = rela_info[()]['obj_attr']
                else:
                    #if we do not have rela_matrix, this matrix is set to be [0,3] zero matrix
                    rela_data = {}
                    rela_data['rela_rela_matrix'] = []
                    rela_data['rela_attr_matrix'] = []
            else:
                rela_data = {}
                rela_data['rela_rela_matrix'] = []
                rela_data['rela_attr_matrix'] = []

        else:
            att_feat = np.zeros((1,1,1))
        return (np.load(os.path.join(self.input_fc_dir, str(self.img_ids[ix]) + '.npy')),
                att_feat,
                rela_data,
                ix)

    def __len__(self):
        return len(self.info['images'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[3] == ix, "ix not equal"

        return tmp + [wrapped]
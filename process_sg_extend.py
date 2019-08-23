import json
import numpy as np
import os
from models.ass_fun import *
root_path = '/home/yangxu/project/self-critical.pytorch/'
# spice_sg_dict_path = root_path + 'data/spice_sg_dict.npz'
spice_sg_dict_path = root_path + 'data/spice_sg_dict2.npz'
# sg_dict_extend_path = root_path + 'data/sg_dict_extend.npz'
sg_dict_extend_path = root_path + 'data/sg_dict_extend2.npz'
rela_dict_path = root_path + 'data/coco_rela_pred_dict.npy'
cocobu_rela_folder = root_path + 'data/coco_pred_sg/'
# cocobu_rela_img_folder = root_path + 'data/cocobu_sg_img/'
cocobu_rela_img_folder = root_path + 'data/cocobu_sg_img2/'

spice_dict = np.load(spice_sg_dict_path)['spice_dict'][()]

sen_word_to_ix = spice_dict['word_to_ix']
sen_ix_to_word = spice_dict['ix_to_word']
sen_words = sen_word_to_ix.keys()
sen_idx_max = len(sen_word_to_ix)

rela_dict_info = np.load(rela_dict_path)[()]
rela_dict = rela_dict_info['rela_dict']

N_rela_dict = len(rela_dict)
for i in range(N_rela_dict):
	word_temp = rela_dict[i]
	if word_temp not in sen_words:
		sen_ix_to_word[sen_idx_max] = word_temp
		sen_word_to_ix[word_temp] = sen_idx_max
		sen_idx_max += 1

sg_dict_extend = {}
sg_dict_extend['sg_word_to_ix'] = sen_word_to_ix
sg_dict_extend['sg_ix_to_word'] = sen_ix_to_word
np.savez(sg_dict_extend_path,sg_dict = sg_dict_extend)

t = 0
for file in os.listdir(cocobu_rela_folder):
	if file.endswith(".npy"):
		t = t+1
		if t%1000 == 0:
			print('processing {0} data'.format(t))
		rela_info = np.load(cocobu_rela_folder+file)[()]
		rela_matrix = rela_info['rela_matrix']
		attr_matrix = rela_info['obj_attr']
		rela_matrix_new = np.zeros(np.shape(rela_matrix))
		attr_matrix_new = np.zeros(np.shape(attr_matrix))

		N_rela = len(rela_matrix)
		for i in range(N_rela):
			rela_matrix_new[i,0] = rela_matrix[i,0]
			rela_matrix_new[i,1] = rela_matrix[i,1]
			rela_temp = rela_dict[int(rela_matrix[i,2])]
			rela_matrix_new[i,2] = sen_word_to_ix[rela_temp]

		N_attr = len(attr_matrix)
		for i in range(N_attr):
			if np.shape(attr_matrix)[1] > 0:
				attr_matrix_new[i,0] = attr_matrix[i,0]
				for j in range(1,np.shape(attr_matrix)[1]):
					if attr_matrix[i,j] != -1:
  						attr_temp = rela_dict[int(attr_matrix[i,j])]
						attr_matrix_new[i,j] = sen_word_to_ix[attr_temp]
					else:
						attr_matrix_new[i,j] = -1
		rela_info_new = {}
		rela_info_new['attr_matrix'] = attr_matrix_new
		rela_info_new['rela_matrix'] = rela_matrix_new

		np.save(cocobu_rela_img_folder + file, rela_info_new)





















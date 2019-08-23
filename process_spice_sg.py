import json
import numpy as np
import os

from models.ass_fun import *
root_path = '/home/yangxu/project/self-critical.pytorch/'
#root_path = '/home/yangsheng/project/self-critical.pytorch/'
train_sg_path = root_path + 'data/spice_sg_train.json'
val_sg_path = root_path + 'data/spice_sg_val.json'
spice_sg_folder = root_path + 'data/coco_spice_sg2/'
spice_sg_dict_raw_path = root_path + 'data/spice_sg_dict_raw2.npz'
spice_sg_dict_path = root_path + 'data/spice_sg_dict2.npz'

dict_info = json.load(open(root_path+'data/cocobu2.json'))
ix_to_word = dict_info['ix_to_word']
ix_max = 0
N_save = 10
dict_spice = {}
dict_num = {}
for ix in ix_to_word.keys():
	dict_spice[ix_to_word[ix]] = int(ix)
	dict_num[ix_to_word[ix]] = N_save+1
	ix_max = max(int(ix), ix_max)
dict_index = ix_max+1

train_sg = json.load(open(train_sg_path))
val_sg = json.load(open(val_sg_path))
train_sg.update(val_sg)
all_sg = train_sg

img_ids = all_sg.keys()

if os.path.isfile(spice_sg_dict_raw_path) == 0:
	print('create a raw dict:')
	t = 0
	for img_id in img_ids:
		t += 1
		if t % 1000 == 0:
			print("processing {0} data".format(t))
			
		sg_temp = all_sg[img_id]
		rela = sg_temp['rela']
		sbj = sg_temp['subject']
		obj = sg_temp['object']
		attr = sg_temp['attribute']
		N_rela = len(rela)
		N_attr = len(attr)
		for i in range(N_rela):
			rela_temp = rela[i].strip()
			sbj_temp = sbj[i].strip()
			obj_temp = obj[i].strip()

			rela_temp = change_word(rela_temp)
			sbj_temp = change_word(sbj_temp)
			obj_temp = change_word(obj_temp)

			all_sg[img_id]['rela'][i] = rela_temp
			all_sg[img_id]['subject'][i] = sbj_temp
			all_sg[img_id]['object'][i] = obj_temp

			if rela_temp not in dict_spice.keys():
				dict_spice[rela_temp] = dict_index
				dict_num[rela_temp] = 1
				dict_index += 1
			else:
				dict_num[rela_temp] += 1

			if sbj_temp not in dict_spice.keys():
				dict_spice[sbj_temp] = dict_index
				dict_num[sbj_temp] = 1
				dict_index += 1
			else:
				dict_num[sbj_temp] += 1

			if obj_temp not in dict_spice.keys():
				dict_spice[obj_temp] = dict_index
				dict_num[obj_temp] = 1
				dict_index += 1
			else:
				dict_num[obj_temp] += 1

		for i in range(N_attr):
			node_temp = attr[i][5:].strip()
			node_temp = change_word(node_temp)
			all_sg[img_id]['attribute'][i] = all_sg[img_id]['attribute'][i][:5] + node_temp

			if node_temp not in dict_spice.keys():
				dict_spice[node_temp] = dict_index
				dict_num[node_temp] = 1
				dict_index += 1
			else:
				dict_num[node_temp] += 1

	print('ix_max is {0}, dict_index is {1}'.format(ix_max, dict_index))

	spice_ix_to_word = {}
	for word in dict_spice.keys():
		spice_ix_to_word[dict_spice[word]] = word
	spice_word_to_ix = dict_spice
	spice_dict= {}
	spice_dict['spice_ix_to_word'] = spice_ix_to_word
	spice_dict['spice_word_to_ix'] = spice_word_to_ix
	spice_dict['dict_num'] = dict_num
	spice_dict_raw = spice_dict
	np.savez(spice_sg_dict_raw_path, spice_dict = spice_dict)
	n = 0
	word_keys = dict_num.keys()
	for word in word_keys:
		if dict_num[word] >= N_save:
			n += 1
	print('number of words larger than N_save'.format(n))
else:
	num_ssg = 0
	print('load raw dict:')
	spice_dict_raw = np.load(spice_sg_dict_raw_path)['spice_dict'][()]
	dict_num_raw = spice_dict_raw['dict_num']
	n = 0
	word_keys = dict_num_raw.keys()
	for word in word_keys:
		if dict_num_raw[word] >= N_save:
			n += 1
		if dict_num_raw[word] > N_save+1:
			num_ssg += 1
	print('number of words larger than N_save is {0}'.format(n))
	print('number of words larger than N_save is {0}'.format(num_ssg))


if os.path.isfile(spice_sg_dict_path) == 0:
	print('create spice sg dict')
	spice_raw_itw = spice_dict_raw['spice_ix_to_word']
	spice_raw_wti = spice_dict_raw['spice_word_to_ix']
	dict_num_raw = spice_dict_raw['dict_num']
	itw_new = {}
	wti_new = {}
	word_keys = dict_num_raw.keys()
	for word in word_keys:
		if dict_num_raw[word] >= N_save:
			wti_new[word] = spice_raw_wti[word]
			itw_new[wti_new[word]] = word
		if dict_num_raw[word] > N_save:
			num_ssg = num_ssg + 1
	wti = {}
	itw = {}

	N_dict = len(itw_new)
	ids = itw_new.keys()
	ids_sort = np.sort(ids)
	for i in range(1,N_dict+1):
		itw[i] = itw_new[ids_sort[i-1]]
		wti[itw[i]] = i
	spice_dict = {}
	spice_dict['ix_to_word'] = itw
	spice_dict['word_to_ix'] = wti
	np.savez(spice_sg_dict_path, spice_dict = spice_dict)
	print('num_ssg: {0}'.format(num_ssg))
else:
	print('load dict:')
	spice_dict = np.load(spice_sg_dict_path)['spice_dict'][()]

wti = spice_dict['word_to_ix']
itw = spice_dict['ix_to_word']
t = 0
for img_id in img_ids:
	t += 1
	if t % 1000 == 0:
		print("processing {0} data".format(t))
		
	sg_temp = all_sg[img_id]
	rela = sg_temp['rela']
	sbj = sg_temp['subject']
	obj = sg_temp['object']
	attr = sg_temp['attribute']



	ids = 0
	N_rela = len(rela)
	N_attr = len(attr)
	rela_matrix = np.zeros([N_rela, 3])
	unique_obj = []
	for i in range(N_rela):
		sbj_temp = sbj[i].strip()
		obj_temp = obj[i].strip()
		rela_temp = rela[i].strip()

		sbj_temp = change_word(sbj_temp)
		obj_temp = change_word(obj_temp)
		rela_temp = change_word(rela_temp)
	
		if sbj_temp in wti.keys():
			unique_obj.append(wti[sbj_temp])
		if obj_temp in wti.keys():
			unique_obj.append(wti[obj_temp])
		if sbj_temp in wti.keys():
			if obj_temp in wti.keys():
				if rela_temp in wti.keys():
					rela_matrix[ids,0] = wti[sbj_temp]
					rela_matrix[ids,1] = wti[obj_temp]
					rela_matrix[ids,2] = wti[rela_temp]
					ids += 1
	N_rela = ids
	rela_matrix = rela_matrix[:N_rela]

	for i in range(N_attr):
		node_temp = attr[i][5:].strip()
		node_temp = change_word(node_temp)
		index = attr[i][:5]
		if index == 'node:':
			if node_temp in wti.keys():
				unique_obj.append(wti[node_temp])

	unique_obj = np.unique(unique_obj)

	obj_info = []
	N_obj = len(unique_obj)
	for i in range(N_obj):
		attr_info = []
		attr_info.append(unique_obj[i])
		for j in range(N_attr):
			if attr[j][:5]== 'node:':
				node_temp = attr[j][5:].strip()
				node_temp = change_word(node_temp)

				if node_temp in wti.keys():
					if wti[node_temp] == unique_obj[i]:
						for k in range(j+1, N_attr):
							if attr[k][:5] == 'node:':
								break
							attr_temp = attr[k][5:].strip()
							attr_temp = change_word(attr_temp)

							if attr_temp in wti.keys():
								attr_info.append(wti[attr_temp])

		obj_info.append(attr_info)

	for i in range(N_rela):
		sbj_index = np.where(unique_obj==rela_matrix[i,0])[0][0]
		obj_index = np.where(unique_obj==rela_matrix[i,1])[0][0]
		rela_matrix[i,0] = sbj_index
		rela_matrix[i,1] = obj_index

	ssg = {}
	ssg['obj_info'] = obj_info
	ssg['rela_info'] = rela_matrix
	save_path_temp = spice_sg_folder + img_id +'.npy'

	if t<=10:
		N_rela = len(rela_matrix)
		for i in range(N_rela):
			sbj_temp = itw[int(obj_info[int(rela_matrix[i,0])][0])]
			obj_temp = itw[int(obj_info[int(rela_matrix[i,1])][0])]
			rela_temp = itw[int(rela_matrix[i,2])]
			print('{0}-{1}-{2}'.format(sbj_temp,rela_temp,obj_temp))
		N_obj = len(obj_info)
		for i in range(N_obj):
			obj_temp = obj_info[i]
			N_attr = len(obj_temp)
			for j in range(N_attr):
				if j!= 0:
					print('--{0}'.format(itw[int(obj_temp[j])]))
				else:
					print(itw[int(obj_temp[j])])

		N_rela = len(sg_temp['rela'])
		for i in range(N_rela):
			print('{0}-{1}-{2}'.format(sg_temp['subject'][i], sg_temp['rela'][i], sg_temp['object'][i]))
		N_attr = len(sg_temp['attribute'])
		for i in range(N_attr):
			print(sg_temp['attribute'][i])
	np.save(save_path_temp, ssg)







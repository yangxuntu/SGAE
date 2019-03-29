import numpy as np
from nltk.stem import WordNetLemmatizer

def read_roidb(roidb_path):
	roidb_file = np.load(roidb_path)
	key = roidb_file.keys()[0]
	roidb_temp = roidb_file[key]
	roidb = roidb_temp[()]
	return roidb

def extract_cor(rela_temp, obj, w_ratio,h_ratio):
	if obj == None:
		h = rela_temp['h']
		w = rela_temp['w']
		y1 = rela_temp['y']
		x1 = rela_temp['x']
	else:
		h = rela_temp[obj]['h']
		w = rela_temp[obj]['w']
		y1 = rela_temp[obj]['y']
		x1 = rela_temp[obj]['x']
	x2 = x1 + w
	y2 = y1 + h

	x1 = x1 * w_ratio
	y1 = y1 * h_ratio
	x2 = x2 * w_ratio
	y2 = y2 * h_ratio
	box = np.zeros(shape=[4,])
	box[0] = x1
	box[1] = y1
	box[2] = x2
	box[3] = y2

	return box

def compute_iou(box, proposal):
	"""
	compute the IoU between box with proposal
	Arg:
		box: [x1,y1,x2,y2]
		proposal: N*4 matrix, each line is [p_x1,p_y1,p_x2,p_y2]
	output:
		IoU: N*1 matrix, every IoU[i] means the IoU between
			 box with proposal[i,:]
	"""
	len_proposal = np.shape(proposal)[0]
	IoU = np.empty([len_proposal,1])
	for i in range(len_proposal):
		xA = max(box[0], proposal[i,0])
		yA = max(box[1], proposal[i,1])
		xB = min(box[2], proposal[i,2])
		yB = min(box[3], proposal[i,3])

		if xB<xA or yB<yA:
			IoU[i,0]=0
		else:
			area_I = (xB - xA) * (yB - yA)
			area1 = (box[2] - box[0])*(box[3] - box[1])
			area2 = (proposal[i,2] - proposal[i,0])*(proposal[i,3] - proposal[i,1])
			IoU[i,0] = area_I/float(area1 + area2 - area_I)
	return IoU

def find_exist_relation(obj_box, sbj_box, cocobu_box_temp, th):
	"""
	find boxes in cocobu_box_temp which own IOU smaller than th with
	obj_box and sbj_box.
	"""
	obj_iou = compute_iou(obj_box, cocobu_box_temp)
	sbj_iou = compute_iou(sbj_box, cocobu_box_temp)
	obj_index = np.where(obj_iou >= th)[0]
	sbj_index = np.where(sbj_iou >= th)[0]
	return obj_index, sbj_index

def change_word(word_ori):
	"""
	Lemmatizer a word, like change 'holding' to 'hold' or
	'cats' to 'cat'
	"""
	word_ori = word_ori.lower()
	lem = WordNetLemmatizer()
	word_change = lem.lemmatize(word_ori)
	if word_change == word_ori:
		word_change = lem.lemmatize(word_ori,'v')
	return word_change


def print_rela_info(rela_use, i2w):
	"""
	print relationship information
	:param rela_use: contains rela_info and obj_info
	:param i2w: dict, i2w[index] is the index-th word in i2w
	"""
	rela_matrix = rela_use['rela_info']
	obj_matrix = rela_use['obj_info']
	N_rela = len(rela_matrix)
	N_obj = len(obj_matrix)

	for i in range(N_rela):
		sbj_id = int(rela_matrix[i, 0])
		sbj = i2w[obj_matrix[sbj_id][0]]
		obj_id = int(rela_matrix[i, 1])
		obj = i2w[obj_matrix[obj_id][0]]
		pred = i2w[rela_matrix[i, 2]]
		print('sbj-pred-obj:{0}-{1}-{2}'.format(sbj,pred,obj))

	for i in range(N_obj):
		obj_list = obj_matrix[i]
		N_obj_temp = len(obj_list)
		for j in range(N_obj_temp):
			word_temp = i2w[obj_list[j]]
			if j!=0:
				print('---{0}'.format(word_temp))
			else:
				print(word_temp)
















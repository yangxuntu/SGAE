from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .OldModel import ShowAttendTellModel, AllImgModel
#from .Att2inModel import Att2inModel
from .AttModel import *
from .AttModel_mapping import *
from .AttModel_et import *
from .AttModel_dt import *
from .AttModel_dt2 import *
from .AttModel_mem import *
from .AttModel_mem2 import *
from .AttModel_mem3 import *
from .AttModel_mem4 import *
from .AttModel_mem5 import *
from .SgModel import *
from .AttModel_gfc import *
from .AttModel_mem4_gfc import *
from .AttModel_mem5_gfc import *
from .AttModel_mem4_new import *
from .AttModel_mem5_new import *
from .AttModel_rs import *
from .AttModel_rs2 import *
from .AttModel_rs3 import *
from .AttModel_rs3_st import *
from .AttModel_rs3_mem import *
from .AttModel_rs3_mem2 import *
from .AttModel_rs4 import *
from .AttModel_rs5 import *
from .AttModel_rs6 import *

def setup(opt):
    
    if opt.caption_model == 'fc':
        model = FCModel(opt)
    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    #elif opt.caption_model == 'att2in2':
        #model = Att2in2Model(opt)
    elif opt.caption_model == 'att2all2':
        model = Att2all2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    elif opt.caption_model == 'mtopdown':
        model = MTopDownModel(opt)
    # StackAtt
    elif opt.caption_model == 'stackatt':
        model = StackAttModel(opt)
    # DenseAtt
    elif opt.caption_model == 'denseatt':
        model = DenseAttModel(opt)
    elif opt.caption_model == 'gtssg':
        model = GTssgModel(opt)
    elif opt.caption_model == 'lstm_map':
        model = LSTM_map(opt)
    elif opt.caption_model == 'lstm_et':
        model = LSTM_et(opt)
    elif opt.caption_model == 'lstm_dt':
        model = LSTM_dt(opt)
    elif opt.caption_model == 'lstm_dt2':
        model = LSTM_dt2(opt)
    elif opt.caption_model == 'lstm_mem':
        model = LSTM_mem(opt)
    elif opt.caption_model == 'lstm_mem2':
        model = LSTM_mem2(opt)
    elif opt.caption_model == 'lstm_mem3':
        model = LSTM_mem3(opt)
    elif opt.caption_model == 'lstm_mem4':
        model = LSTM_mem4(opt)
    elif opt.caption_model == 'lstm_mem5':
        model = LSTM_mem5(opt)
    elif opt.caption_model == 'sg':
        model = SgModel(opt)
    elif opt.caption_model == 'att_gfc':
        model = TopDownModel_gfc(opt)
    elif opt.caption_model == 'lstm_mem4_gfc':
        model = LSTM_mem4_gfc(opt)
    elif opt.caption_model == 'lstm_mem5_gfc':
        model = LSTM_mem5_gfc(opt)
    elif opt.caption_model == 'lstm_mem4_new':
        model = LSTM_mem4_new(opt)
    elif opt.caption_model == 'lstm_mem5_new':
        model = LSTM_mem5_new(opt)
    elif opt.caption_model == 'cap_rs':
        model = Cap_Reason(opt)
    elif opt.caption_model == 'cap_rs2':
        model = Cap_Reason2(opt)
    elif opt.caption_model == 'cap_rs3':
        model = Cap_Reason3(opt)
    elif opt.caption_model == 'cap_rs3_st':
        model = Cap_Reason3_st(opt)
    elif opt.caption_model == 'mcap_rs3_st':
        model = MCap_Reason3_st(opt)
    elif opt.caption_model == 'cap_rs3_mem':
        model = Cap_Reason3_mem(opt)
    elif opt.caption_model == 'mcap_rs3_mem':
        model = MCap_Reason3_mem(opt)
    elif opt.caption_model == 'mcap_rs3_mem2':
        model = MCap_Reason3_mem2(opt)
    elif opt.caption_model == 'mcap_rs3':
        model = MCap_Reason3(opt)
    elif opt.caption_model == 'cap_rs4':
        model = Cap_Reason4(opt)
    elif opt.caption_model == 'cap_rs5':
        model = Cap_Reason5(opt)
    elif opt.caption_model == 'cap_rs6':
        model = Cap_Reason6(opt)
    elif opt.caption_model == 'mcap_rs6':
        model = MCap_Reason6(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    save_id_real = getattr(opt, 'save_id', '')
    if save_id_real == '':
        save_id_real = opt.id

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.checkpoint_path)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.checkpoint_path, 'infos_'
                + save_id_real + format(int(opt.start_from),'04') + '.pkl')),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(
            opt.checkpoint_path, 'model' +save_id_real+ format(int(opt.start_from),'04') + '.pth')))

    return model
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/coco.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/cocotalk_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/cocotalk_att',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='data/cocotalk_box',
                    help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_attr_dir', type=str,
                        help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_rela_dir', type=str,
                        help='path to the directory containing the boxes of att feats')
    #parser.add_argument('--input_rela_dir', type=str, default='data/cocotalk_rela',
                        #help='path to the directory containing the relationships of att feats')
    parser.add_argument('--input_sence_dir', type=str, default='data/cocobu_sence',
                        help='path to the directory containing the relationships of att feats')
    parser.add_argument('--input_ssg_dir', type=str, default='data/gt_sg',
                        help='path to the directory containing the ground truth sentence scene graph')
    parser.add_argument('--input_isg_dir', type=str, default='data/gt_sg',
                        help='path to the directory containing the image scene graph')
    parser.add_argument('--input_label_h5', type=str, default='data/coco_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_coco_sg_rela', type=str, default='data/coco_sg_rela.npy',
                        help="path to the coco's scene graph' relationship information" )

    parser.add_argument('--rela_dict_dir', type=str, default='data/rela_dict.npy',
                        help='path to the npy file contains rela dict info')
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')
    parser.add_argument('--train_split', type=str, default='train',
                        help='which split used to train')

    # Model settings
    parser.add_argument('--sg_model', type=str, default="",
                        help='sg model path')
    parser.add_argument('--caption_model', type=str, default="show_tell",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, att2all2, adaatt, adaattmo, topdown, stackatt, denseatt, gtssg')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--gru_t', type=int, default=4,
                        help='the numbers of gru will iterate')
    parser.add_argument('--lstm_sim_loss', type=str, default='l2',
                        help='the loss which used to make sentence and image hidden state to be equal')
    parser.add_argument('--sim_lambda', type=float, default=1,
                        help='the balance parameter used for making hidden state to be equal')
    parser.add_argument('--save_id', type=str, default='',
                        help='save id')
    parser.add_argument('--topdown_res', type=int, default=0,
                        help='whether use residual connection in topdown model')
    parser.add_argument('--mtopdown_res', type=int, default=0,
                        help='whether use residual connection in mtopdown model')
    parser.add_argument('--mtopdown_num', type=int, default=1,
                        help='the number of blocks topdown moduel will be used')


    parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')

    # feature manipulation
    parser.add_argument('--norm_att_feat', type=int, default=0,
                    help='If normalize attention features')
    parser.add_argument('--use_box', type=int, default=0,
                    help='If use box features')
    parser.add_argument('--use_obj', type=int, default=0,
                        help='If learn object information when training scene graph classification')
    parser.add_argument('--use_attr', type=int, default=0,
                        help='If learn attribute information when training scene graph classification')
    parser.add_argument('--use_rela', type=int, default=0,
                        help='If use rela information')
    parser.add_argument('--use_gru', type=int, default=0,
                        help='If use gru')
    parser.add_argument('--use_gfc', type=int, default=0,
                        help='If use gfc')
    parser.add_argument('--use_ssg', type=int, default=0,
                        help='If use ssg')
    parser.add_argument('--use_isg', type=int, default=0,
                        help='If use isg')
    parser.add_argument('--use_attr_info', type=int, default=1,
                        help='If use attributes info')
    parser.add_argument('--sg_train_mode', type=str, default='rela',
                        help='which scene graph info will be trained')
    parser.add_argument('--ssg_dict_path', type=str, default='data/ssg_dict.npz',
                        help='path to the sentence scene graph directory')
    parser.add_argument('--sg_index_path', type=str, default='data/spice_sg_index.npz',
                        help='path to the spice_sg_index')
    parser.add_argument('--sg_dict_path', type=str, default='data/sg_dict.npz',
                        help='path to the combined scene graph directory')
    parser.add_argument('--norm_box_feat', type=int, default=0,
                    help='If use box, do we normalize box feature')
    parser.add_argument('--step2_train_after', type=int, default=10,
                        help='when step two trianing begins')
    parser.add_argument('--step3_train_after', type=int, default=20,
                        help='when step two trianing begins')
    parser.add_argument('--step4_train_after', type=int, default=30,
                        help='when step three trianing begins')
    parser.add_argument('--sen_img_equal_epoch', type=int, default=10,
                        help='when to make sen sg and img sg equal')
    parser.add_argument('--train_img2sen_epoch', type=int, default=10,
                        help='when step two trianing begins')
    parser.add_argument('--which_to_extract', type=str, default='e',
                        help='which data is extracted, e means embedding, h means hidden state')
    parser.add_argument('--memory_cell_path', type=str, default='0',
                        help='memory cell path')
    parser.add_argument('--senti_coco', type=int, default=0,
                        help='whether go to do sentiment caption task')
    parser.add_argument('--senti_dict_path', type=str, default='0',
                        help='the path to sentiment dict')
    parser.add_argument('--senti_data_path', type=str, default='0',
                        help='the path to sentiment data')
    parser.add_argument('--senti_attitude', type=int, default=1,
                        help='1 means positive, 0 means negative')
    parser.add_argument('--rbm_logit', type=int, default='0',
                        help='whether use rbm')
    parser.add_argument('--rbm_size', type=int, default='2000',
                        help='rbm_size')
    parser.add_argument('--gpu', type=int, default='0',
                        help='gpu_id')
    parser.add_argument('--pretrain_model', type=str, default='0',
                        help='pretraind model')
    parser.add_argument('--sg_ft', type=int, default='1',
                        help='whether finetune sg net')
    parser.add_argument('--combine_att', type=str, default='add',
                        help='how to combine att feats: add or concatenate')
    parser.add_argument('--cont_ver', type=int, default='0',
                        help='which kind of controller is used, 0 means no controller is used')
    parser.add_argument('--sg_net_index', type=int, default='1',
                        help='use sg net')
    parser.add_argument('--rela_mod_index', type=int, default='1',
                        help='use rela mod')
    parser.add_argument('--rela_mod_layer', type=int, default='2',
                        help='self att layer')
    parser.add_argument('--gum_soft', type=int, default='0',
                        help='whether gumble softmax')


    parser.add_argument('--memory_index', type=str, default='h',
                        help='which memory is used')
    parser.add_argument('--memory_size', type=int, default=1000,
                        help='how much is the memory size')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_rl', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_every_rl', type=int, default=3,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate_rl', type=float, default=0.8,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')
    parser.add_argument('--accumulate_number', type=int, default=1,
                        help='how many times it should accumulate the gradients, the truth batch_size=accumulate_number*batch_size')
    parser.add_argument('--relu_mod', type=str, default='relu',
                        help='relu_mod')
    parser.add_argument('--leaky_relu_value', type=float, default=0.1,
                        help='relu_mod')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=10,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       

    # misc
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')

    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1,
                    help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0,
                    help='The reward weight from bleu4')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args
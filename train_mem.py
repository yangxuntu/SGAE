from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader_extend import *
import eval_utils_mem
import misc.utils as utils
from misc.rewards_mem import init_scorer, get_self_critical_reward

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    training_mode = 0
    optimizer_reset = 0
    change_mode1 = 0
    change_mode2 = 0

    use_rela = getattr(opt,'use_rela',0)
    if use_rela:
        opt.rela_dict_size = loader.rela_dict_size
    #need another parameter to control how to train the model

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + format(int(opt.start_from),'04') + '.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + format(int(opt.start_from),'04') + '.pkl')):
            with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + format(int(opt.start_from),'04') + '.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    if epoch >= opt.step2_train_after and epoch < opt.step3_train_after:
        training_mode = 1
    elif epoch >= opt.step3_train_after:
        training_mode = 2
    else:
        training_mode = 0

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt).cuda()
    #dp_model = torch.nn.DataParallel(model)
    #dp_model = torch.nn.DataParallel(model, [0, 1])
    dp_model = model
    for name, param in model.named_parameters():
        print(name)

    update_lr_flag = True
    # Assure in training mode
    dp_model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    optimizer = utils.build_optimizer(model.parameters(), opt)
    optimizer_mem = optim.Adam([model.memory_cell], opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon,
               weight_decay=opt.weight_decay)

    # Load the optimizer

    if vars(opt).get('start_from', None) is not None and os.path.isfile(
            os.path.join(opt.checkpoint_path, 'optimizer' + opt.id + format(int(opt.start_from),'04')+'.pth')):
        optimizer.load_state_dict(torch.load(os.path.join(
            opt.checkpoint_path, 'optimizer' + opt.id + format(int(opt.start_from),'04')+'.pth')))
        if (training_mode == 1 or training_mode == 2) and os.path.isfile(
            os.path.join(opt.checkpoint_path, 'optimizer_mem' + opt.id + format(int(opt.start_from),'04')+'.pth')):
                optimizer_mem.load_state_dict(torch.load(os.path.join(
                    opt.checkpoint_path, 'optimizer_mem' + opt.id + format(int(opt.start_from), '04') + '.pth')))

    optimizer.zero_grad()
    optimizer_mem.zero_grad()
    accumulate_iter = 0
    reward = np.zeros([1,1])
    train_loss = 0

    while True:
        # if optimizer_reset == 1:
        #     print("++++++++++++++++++++++++++++++")
        #     print('reset optimizer')
        #     print("++++++++++++++++++++++++++++++")
        #     optimizer = utils.build_optimizer(model.parameters(), opt)
        #     optimizer_mem = optim.Adam([model.memory_cell], opt.learning_rate, (opt.optim_alpha, opt.optim_beta),
        #                                opt.optim_epsilon,
        #                                weight_decay=opt.weight_decay)
        #     optimizer_reset = 0


        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False
                
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch(opt.train_split)
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        if epoch >= opt.step2_train_after and epoch < opt.step3_train_after:
            training_mode = 1
            if change_mode1 == 0:
                change_mode1 = 1
                optimizer_reset = 1
        elif epoch >= opt.step3_train_after:
            training_mode = 2
            if change_mode2 == 0:
                change_mode2 = 1
                optimizer_reset = 1
        else:
            training_mode = 0

        fc_feats = None
        att_feats = None
        att_masks = None
        ssg_data = None
        rela_data = None

        tmp = [data['fc_feats'], data['labels'], data['masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, labels, masks = tmp


        tmp = [data['att_feats'], data['att_masks'],data['rela_rela_matrix'],
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

        if not sc_flag:
            loss = crit(dp_model(fc_feats, att_feats, labels, att_masks, rela_data, ssg_data, use_rela, training_mode),
                        labels[:, 1:], masks[:, 1:])
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, rela_data, ssg_data,
                                                   use_rela, training_mode, opt={'sample_max':0}, mode='sample')

            rela_data = {}
            rela_data['att_feats'] = att_feats
            rela_data['att_masks'] = att_masks
            rela_data['rela_matrix'] = rela_rela_matrix
            rela_data['rela_masks'] = rela_rela_masks
            rela_data['attr_matrix'] = rela_attr_matrix
            rela_data['attr_masks'] = rela_attr_masks

            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, rela_data, ssg_data,
                                                   use_rela, training_mode, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        accumulate_iter = accumulate_iter + 1
        loss = loss/opt.accumulate_number
        loss.backward()

        if accumulate_iter % opt.accumulate_number == 0:
            if training_mode == 0 :
                utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            elif training_mode == 1:
                utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                utils.clip_gradient(optimizer_mem, opt.grad_clip)
                optimizer_mem.step()
                optimizer_mem.zero_grad()
            elif training_mode == 2:
                utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                utils.clip_gradient(optimizer_mem, opt.grad_clip)
                optimizer_mem.step()
                optimizer_mem.zero_grad()

            iteration += 1
            accumulate_iter = 0
            train_loss = loss.item() * opt.accumulate_number
            end = time.time()
            text_file = open(opt.id+'.txt', "aw")
            if not sc_flag:
                print("iter {} (epoch {}), train_model {}, train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, training_mode, train_loss, end - start))
                text_file.write("iter {} (epoch {}), train_model {}, train_loss = {:.3f}, time/batch = {:.3f}\n" \
                      .format(iteration, epoch, training_mode, train_loss, end - start))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, np.mean(reward[:, 0]), end - start))
                text_file.write("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}\n" \
                      .format(iteration, epoch, np.mean(reward[:, 0]), end - start))
            text_file.close()

        torch.cuda.synchronize()

        # Update the iteration and epoch

        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0) and (accumulate_iter % opt.accumulate_number == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0) and (accumulate_iter % opt.accumulate_number == 0):
            # eval model

            eval_kwargs = {'split': 'test',
                            'dataset': opt.input_json,
                            'use_rela': use_rela,
                           'num_images': 1,
                           }
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils_mem.eval_split(dp_model, crit, loader, training_mode, eval_kwargs)

            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                save_id = iteration/opt.save_checkpoint_every
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model'+opt.id+format(int(save_id),'04')+'.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer'+opt.id+format(int(save_id),'04')+'.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                if training_mode == 1 or training_mode == 2 or opt.caption_model == 'lstm_mem':
                    optimizer_mem_path = os.path.join(opt.checkpoint_path,
                                              'optimizer_mem' + opt.id + format(int(save_id), '04') + '.pth')
                    torch.save(optimizer_mem.state_dict(), optimizer_mem_path)

                    memory_cell = dp_model.memory_cell.data.cpu().numpy()
                    memory_cell_path = os.path.join(opt.checkpoint_path,
                                              'memory_cell' + opt.id + format(int(save_id), '04') + '.npz')
                    np.savez(memory_cell_path, memory_cell=memory_cell)




                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+format(int(save_id),'04')+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+format(int(save_id),'04')+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

os.environ["CUDA_VISIBLE_DEVICES"]="1"
opt = opts.parse_opt()
train(opt)

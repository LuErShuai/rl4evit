# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torchvision.transforms as transforms
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from collections import namedtuple
import time
import random 
import numpy as np
from tensorboardX import SummaryWriter
from pathlib import Path
import json
from PIL import Image

timestamp = time.time()
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(timestamp))
writer = SummaryWriter('./runs/Agent/reward_{}'.format(formatted_time))
sample_num = 0
max_accuracy = 0
token_depth_global = 0
eval_num = 0

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, data_loader_val: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5
    # model.agent.reward_one_epoch = 0
    global sample_num
    
    batch_num = 1
    keep_ratio = [0,0,0]
    # torch.cuda.empty_cache()
    for samples, targets, paths in metric_logger.log_every(data_loader, print_freq, header):
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    # for samples, targets, paths in data_loader:
    # for samples, targets in data_loader:
        keep_ratio_batch = [0,0,0]
        token_depth_batch  = 0

        start = time.perf_counter()
        sample_num = sample_num + 1

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
                    
        with torch.cuda.amp.autocast():
            outputs, action_n = model(samples)
            loss = criterion(samples, outputs, targets)
            # outputs_base = model_base(samples)
        

        # size of loss:[batch_size]
        loss_value = loss.item()

        # batch_num=0
        torch.cuda.empty_cache()
        if args.train_agent:
            torch.cuda.empty_cache()
            end_1 = time.perf_counter()
            # train agent
            # classify_results = outputs - targets
            _, outputs_max_index = outputs.max(dim=1)
            _, targets_max_index = targets.max(dim=1)
            # _, outputs_base_max_index = outputs_base.max(dim=1)
            # self.buffer = 
            # {
            #     "state":[], -> [block_num, batch_size, token_num, token_dim]
            #     "state_next":[], 
            #     "action":[],
            #     "action_prob":[]
            # }
            Transition = namedtuple('Transition', ['episode_num','episode_step',
                                                   'obs_n', 'v_n',
                                                   'obs_n_', 'v_n_', 'a_n', 'a_logprob_n',
                                                   'r_n', 'done_n', 'died_win',
                                                   'done_episode', 'cls_token'])   
            # shape of buffer

            # self.buffer = {
            #     "state_n":[], #[block_num, batch_size, token_num, token_dim]
            #     "state_next_n":[],
            #     "cls_token":[],
            #     "action_n":[],
            #     "action_prob_n":[],
            #     "mask":[], #[block_num, batch_size, token_num]
            #     "token_keep_ratio":[]
            # }
            buffers = model.buffer

            # state_n = np.array(buffers["state_n"])
            # state_next_n = np.array(buffers["state_next_n"])
            # died = np.array(buffers["done_n"])
            # new_column = np.ones((1,died.shape[1],died.shape[2]), dtype=died.dtype) 
            # died_with_ones_ = np.concatenate((new_column, died), axis=0)
            # died_with_ones = died_with_ones_[:died.shape[0],:,:]
            
            # shape of died: [3, 64, 197]
            # shape of new_column: [1, 64, 197]
            # died_ = np.array(buffers["done_n"])

            state_n_ = torch.stack(buffers["state_n"])
            v_n_ = torch.stack(buffers["v_n"])
            state_next_n_ = torch.stack(buffers["state_next_n"])
            v_next_n_ = torch.stack(buffers["v_next_n"])
            cls_token_ = torch.stack(buffers["cls_token"])
            action_n_ = torch.stack(buffers["action_n"])
            action_prob_n_ = torch.stack(buffers["action_prob_n"])
            mask_ = torch.stack(buffers["mask"])
            done_n_ = torch.stack(buffers["done_n"])
            died_ = torch.stack(buffers["done_n"])


            # zero out observation for died agent according to mask
            # if agent i died, then there is no state_next_n in current step
            # if agent i died, then there is no state_n in the next step
            # 1 in done_n_ means agent died
            # 0 in done_n_ means agent alive

            new_column = torch.zeros((1, done_n_.shape[1], done_n_.shape[2]),
                                    device=done_n_.device,dtype=done_n_.dtype)
            done_n_with_zeros = torch.cat((new_column, done_n_), axis=0)
            done_n_with_zeros_ = done_n_with_zeros[:done_n_.shape[0],:,:]
        
            state_n_[done_n_with_zeros_==1] = 0
            state_next_n_[done_n_==1] = 0

            batch_size = buffers["state_n"][0].shape[0]
            # episode_step  = buffers["state_n"][0].shape[1]
            # block_num  = len(buffers["state_n"])
            episode_step = len(buffers["state_n"])
            token_keep_ratio = buffers["token_keep_ratio"][0]
            # token_keep_ratio = 0
            
            batch_reward = 0.0

            if args.plot_mask:
                plot_mask(batch_num ,paths, mask_)

            for i in range(batch_size):
                # if vit classify wrong, abandon this trajectory
                # if outputs_base_max_index[i] != targets_max_index[i]:
                #     continue

                # tell if dvit classify correctly
                if outputs_max_index[i] == targets_max_index[i]:
                    classify_correct = True 
                    # batch_num+=1
                else:
                    classify_correct = False

                # done_image = done_n_[:,i,:]
                # keep = torch.unique(done_image, return_counts=True)
                # a = keep[1][0]
                # b = keep[1][0]+keep[1][1]
                # keep_ratio = keep[1][0]/(keep[1][0]+keep[1][1])
                token_depth = 0
                a = 3*196
                b = 3*torch.count_nonzero(action_n_[0,i,:]).item()
                c = 3*torch.count_nonzero(action_n_[1,i,:]).item()
                d = 3*torch.count_nonzero(action_n_[2,i,:]).item()
                token_depth = a+b+c+d
                token_depth_batch += token_depth

                keep_ratio[0] = torch.count_nonzero(action_n_[0,i,:]).item()/196
                keep_ratio[1] = torch.count_nonzero(action_n_[1,i,:]).item()/196
                keep_ratio[2] = torch.count_nonzero(action_n_[2,i,:]).item()/196
                keep_ratio_batch[0] += torch.count_nonzero(action_n_[0,i,:]).item()/196
                keep_ratio_batch[1] += torch.count_nonzero(action_n_[1,i,:]).item()/196
                keep_ratio_batch[2] += torch.count_nonzero(action_n_[2,i,:]).item()/196
                token_keep_ratio = token_depth/(12*196)

                for j in range(episode_step):
                    state_n = state_n_[j][i]
                    state_next_n = state_next_n_[j][i]
                    cls_token = cls_token_[j][i]
                    action_n = action_n_[j][i]
                    action_prob_n = action_prob_n_[j][i]
                    # mask = mask_[j][i]
                    done_n = done_n_[j][i]
                    done_n_last = done_n_[2][i]
                    
                    if j == 2:
                        done_episode = torch.ones(done_n.shape)
                        died_win = torch.ones(done_n.shape)
                    else:
                        done_episode = done_n
                        died_win = done_n
                    # keep = torch.unique(done_n, return_counts=True)

                    reward_n = caculate_reward_per_image(classify_correct,j,
                                                         done_n, keep_ratio,
                                                         token_keep_ratio,
                                                         token_depth)

                    # batch_reward[torch.isnan(batch_reward)] = 0
                    reward_n[torch.isnan(reward_n)] = 0

                    batch_reward += reward_n.sum()
                    # batch_reward[torch.isnan(batch_reward)] = 0

                    v_n = v_n_[j][i]
                    v_next_n = v_next_n_[j][i]

                    # all information include 196 tokens
                    trans = Transition(i,j, state_n, v_n, state_next_n, v_next_n,
                                       action_n, action_prob_n, reward_n,done_n,
                                       died_win, done_episode,cls_token)
                    model.replay_buffer.store_transition(trans)
                model.replay_buffer.episode_num += 1
                if model.replay_buffer.episode_num == 64:
                    break


            # print(model.replay_buffer.episode_num)
            if model.replay_buffer.episode_num >= 64:
                model.agent_n.train(model.replay_buffer, model.replay_buffer.total_step)
                print('batch_reward:', batch_reward/64)
                writer.add_scalar('batch_reward', batch_reward/64, global_step=sample_num)
                writer.add_scalar('token_keep_ratio', token_keep_ratio, global_step=sample_num)
                model.replay_buffer.reset_buffer()

            # if utils.is_main_process() and model.agent.training_step > 50000:
            # if sample_num%100 == 0:
            #     model.agent.save_param()
            #     print(model.agent.total_steps)
            #     print("-------------------save ppo weight-------------------")
            #     # return

        # global max_accuracy
        # if batch_num%100 == 0:
        global max_accuracy
        if batch_num%200 == 0:
        # if batch_num%30 == 29:
            param_path = "epoch" + str(epoch) + "_" + "batch" + str(batch_num)
            model.agent_n.save_agent_n_(param_path)

            test_stat = evaluate_(data_loader_val, model, device, batch_num, args)
            acc_1 = test_stat["acc1"]
            acc_5 = test_stat["acc5"]
            writer.add_scalar('acc_1', acc_1, global_step=int(batch_num/500))
            # if max_accuracy < acc_1:
            #     max_accuracy = acc_1
            #     writer.add_scalar('acc_1', acc_1, global_step=int(batch_num/500))
            #     model.agent_n.save_agent_n()

            # log_stats = {
            #          **{f'test_{k}': v for k, v in test_stat.items()},
            #          'batch_num': batch_num,
            #          'keep_ratio_batch': [x/batch_size for x in keep_ratio_batch],
            #          'token_depth_batch': token_depth_batch,
            #         }
        
            # output_dir = Path(args.output_dir)
            # if args.output_dir and utils.is_main_process():
            #     with (output_dir / "log.txt").open("a") as f:
            #         f.write(json.dumps(log_stats) + "\n")

        if args.train_agent:
            batch_num += 1


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if args.fine_tune:
            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)

        # optimizer.zero_grad()

        # # this attribute is added by timm on one optimizer (adahessian)
        # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.meters['reward_batch'].update(reward_one_batch, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def caculate_reward_per_image(classify_correct, episode_step, done_n,
                              keep_ratio, token_keep_ratio,token_depth):

    # keep = torch.unique(done_n, return_counts=True)
    # keep_num = done_n.numel() - done_n.sum()
    # died_num = done_n.sum()
    # keep_num_ = done_n.numel()
    # keep_ratio = [0.8, 0.8*0.8, 0.8*0.8*0.8]
    # keep_ratio_ = keep_num/keep_num_

    # agent_keep_correct = -1
    # agent_died_correct = 1
    # agent_keep_wrong = 1
    # agent_died_wrong = -1

    # agents_correct = 1
    # agents_wrong = -1
    # alpha = 0.1
    # beta = 1
    # if classify_correct:
    #     reward_1 = (keep_num*agent_keep_correct + died_num*agent_died_correct)
    # else:
    #     reward_1 = 0
    #     # reward_1 = (keep_num*agent_keep_wrong + died_num*agent_died_wrong)

    # if episode_step == 2:
    #     if classify_correct:
    #         reward_2 = agents_correct
    #     else:
    #         reward_2 = agents_wrong
    # else:
    #     reward_2 = 0
    #     

    # reward = alpha*reward_1 + beta*reward_2
    # reward = alpha*reward_1



# def caculate_reward_per_image(classify_correct, episode_step, done_n_last,
#                               keep_ratio_):
# 
#     done_n = done_n_last
#     keep = torch.unique(done_n, return_counts=True)
#     keep_num = done_n.numel() - done_n.sum()
#     keep_num_ = done_n.numel()
#     keep_ratio = [0.8, 0.8*0.8, 0.8*0.8*0.8]
#     keep_ratio_ = keep_num/keep_num_
    # reward_2 = -math.exp(abs(keep_ratio - keep_ratio_))

    # reward_for_classify = 2
    #if classify_correct:
    #    reward_1 = 1.0
    #else:
    #    # reward_1 = -1.0 * reward_for_classify
    #    # reward_1 = -1
    #    # reward_1 = -1
    #    reward_1 = 0

    # reward_2 = keep_ratio_
    # reward_2 = 2-math.exp(abs(keep_ratio[episode_step] - keep_ratio_))
    # reward_2 = 0.5*(keep_ratio[episode_step] - keep_ratio_)
    # reward_2 = 25*(math.exp((1-token_keep_ratio)) - 1) - 4*math.exp(token_keep_ratio)
    # if abs(keep_ratio[episode_step] - keep_ratio_) < 0.05:
    #     reward_2 = 1.0
    # else:
    #     reward_2 = -1.0
    # died_num = done_n.sum()
    # reward_1 = keep_num/196
    # reward_2 = died_num/196



    # if classify_correct:
    #     reward_1 = 0.5
    # else:
    #     # reward_3 = -0.25
    #     # reward_3 = -0.5
    #     reward_3 = 0


    # batch_reward = 0 ，导致没有进行训练
    # if classify_correct:
    #     reward_1 = 1.0 * reward_for_classify
    #     if abs(keep_ratio[episode_step] - keep_ratio_) < 0.1:
    #         reward_2 = 1.0
    #     else:
    #         reward_2 = -1.0
    # else:
    #     # reward_1 = -1.0 * reward_for_classify
    #     reward_1 = 0
    #     reward_2 = 0

    # keep_ratio = keep_num/196
    # reward_4 = 2 - (math.exp(abs(keep_ratio - 0.80)))

    # alpha = 0.5 
    # reward = alpha*reward_1 + (1-alpha)*reward_2
    # reward = reward_1
    # reward = reward_1 * reward_2
    # reward = reward_2
    # reward = reward_2 - reward_1 + reward_3
    # reward = 0.5*reward_4 + reward_3
    # reward = reward_4
    # 奖励为正，则增加保留的token数
    # 奖励为负，则减少保留的token数
    # 这个现象是否正确，背后的原因是什么？
    # 正确的现象应为算法去追逐奖励最大化，从而导致所有agent迅速死亡
    # 所有的agent在一个epoch之后确实都死亡了

    
    # eta=1
    # temp = token_keep_ratio - 0.7
    # if temp < 0:
    #     reward_1 = 1.2 - math.exp(eta*abs(token_keep_ratio - 0.7))
    # elif 0 < temp < 0.1:
    #     reward_1 = 2
    # elif 0.1 < temp:
    #     reward_1 = -1


    # reward = 1 - math.exp(eta*abs(token_keep_ratio - 0.8))
    # reward = 1.5 - math.exp(eta*abs(token_keep_ratio - 0.7))
    # reward = -abs(token_keep_ratio-0.7)
    alive = 1-done_n
    # reward = -0.01*alive.sum()
    # reward = 0.1*done_n.sum()
    # reward = -0.01*done_n.sum()
    

    # if classify_correct:
    #     reward_2 = 3
    # else:
    #     reward_2 = -1
    # done_n[done_n == 0] = -1
    # reward_1 = -0.1
    reward_1 = -0.1*torch.zeros_like(done_n, dtype=torch.long)
    if classify_correct:
        # reward_1 = done_n
        reward_1 = torch.ones_like(done_n, dtype=torch.long)

    reward_2 = done_n
    reward_3 = 1-done_n
    # reward_4 = (1 - token_keep_ratio)*torch.ones_like(done_n, dtype=torch.long)
    # reward_4 = 0.3*(3-episode_step)*(1-keep_ratio[episode_step])*torch.ones_like(done_n,dtype=torch.long)
    keep_ratio_ = [0,0,0]
    keep_ratio_[0] = keep_ratio[0]
    keep_ratio_[1] = keep_ratio[0] - keep_ratio[1]
    keep_ratio_[2] = keep_ratio[1] - keep_ratio[2]
    reward_4 = (1-keep_ratio_[episode_step])*torch.ones_like(done_n,dtype=torch.long)

    keep_ratio_temp = [0.7, 0.7*0.7, 0.7*0.7*0.7]
    # keep_ratio = [0.8, 0.8*0.8, 0.8*0.8*0.8]
    # keep_ratio = [0.4, 0.4*0.4, 0.4*0.4*0.4]
    reward_5 = 0.1-abs((keep_ratio_temp[episode_step] - keep_ratio[episode_step])*torch.ones_like(done_n,dtype=done_n.dtype))

    delta = keep_ratio[episode_step] - keep_ratio_temp[episode_step]
    if delta >= 0:
        reward_6 = done_n
    else:
        reward_6 = 1 - done_n
    eta=1
    beta = 0.01
    
    died_num = done_n.sum()
    alive_num = (1-done_n).sum()
    # reward_7 = -0.106*(1-done_n)
    # reward_7 = -0.031*(1-done_n)
    reward_7 = -0.1*(1-done_n)
    reward_8 = (1-done_n)*torch.zeros_like(done_n, dtype=done_n.dtype)
    if classify_correct:
        reward_8 = (1-done_n)*(24/alive_num)
        # reward_8 = (24/alive_num)
        # reward_8 = (1-done_n)*(22/alive_num)

    # delta = token_depth/2352 - 0.7
    delta = (-keep_ratio_temp[episode_step] + keep_ratio[episode_step])
    if delta > 0:
        reward_9 = -math.exp(delta)*(1-done_n)
    else:
        reward_9 = -math.exp(-delta)*done_n

    reward_10 = 1 - math.exp(abs(delta))
    # reward = eta*reward_1 + beta*reward_2
    # reward = eta*reward_1 - beta*reward_3
    # reward = eta*reward_1 + beta*reward_4/(episode_step + 1)

    # reward = 0.3 - beta*reward_3
    # if not classify_correct:
    #     reward = torch.zeros_like(done_n, dtype=torch.float32)
    # return 1-done_n
    # return reward_1*torch.ones_like(done_n, dtype=torch.float32)
    # return 1.0*reward_7 + 1.0*reward_8
    return 1.0*reward_8 + 1.0*reward_9

    
    


def caculate_reward_per_step(num_block, classify_correct, action, token_keep_ratio,
                             total_steps):
    reward_for_classify = 1
    reward_for_action = 2

    # simplest: split equally
    if classify_correct:
        reward_1 = 1.0*reward_for_classify

        # reward_2 = (1 - action)*2.5*(12-num_block)
        # reward_3 = 0
        # reward_3 = -action*num_block*0.125
        # reward_1 = reward_for_classify/12
        # reward_2 = reward_for_action
        
        # reward_3 = (1 - action)*0.1
    else:
        # reward_1 = -reward_for_classify/12
        # reward_2 = - reward_for_action
        # reward_1 = 0
        reward_1 = -1*reward_for_classify
        # reward_2 = 0
        # reward_3 = 0
        # reward_3 = -(1 - action)*0.1

    # reward_2 = (1 - action)*16*(12-num_block)
    if classify_correct:
        # reward_2 = (1 - action)
        if action == 1:
            reward_2 = 0
        if action == 0:
            reward_2 = 1.0
    else:
        if action == 1:
            # reward_2 = -0.5
            reward_2 = 0
        if action == 0:
            reward_2 = 0
    # reward_2 = (1 - action)*100*(12-num_block)
    # reward_2 = 1 - action
    # reward_3 = -action*num_block*0.0125
    reward_3 = 0

    
    eta = 32
    # reward_4 = - math.exp(eta*abs(token_keep_ratio - 0.7))
    
    # d = token_keep_ratio - 0.75
    # if d > 0:
    #     reward_4 = - action*math.exp(eta*abs(d))
    # if d <= 0:
    #     reward_4 = - (1-action)*math.exp(eta*abs(d))

    reward_2 = 25*(math.exp((1-token_keep_ratio)) - 1) - 4*math.exp(token_keep_ratio)
    # reward_2 = 0
    
    # if token_keep_ratio > 0.75:
    #     reward_4 = -2*action*math.exp(eta*abs(token_keep_ratio-0.75))
    # elif token_keep_ratio <= 0.75 and token_keep_ratio > 0.55:
    #     reward_4 = 2 - math.exp(10*abs(token_keep_ratio - 0.65))
    # elif token_keep_ratio <= 0.55:
    #     reward_4 = -2*(1-action)*math.exp(eta*abs(token_keep_ratio - 0.55))

    if total_steps < 30000:
        if token_keep_ratio > 0.75:
            reward_4 = -2-2*action*math.exp(eta*abs(token_keep_ratio-0.75))
        elif token_keep_ratio <= 0.75 and token_keep_ratio > 0.70:
            reward_4 = -2*action*math.exp(eta*abs(token_keep_ratio-0.7))
        elif token_keep_ratio <= 0.70 and token_keep_ratio > 0.60:
            reward_4 = (1-token_keep_ratio)*1.5
        elif token_keep_ratio <= 0.60 and token_keep_ratio > 0.55:
            reward_4 = -2*(1-action)*math.exp(eta*abs(token_keep_ratio - 0.6))
        elif token_keep_ratio <= 0.55:
            reward_4 = -2-2*(1-action)*math.exp(eta*abs(token_keep_ratio - 0.55))
    else:
        # if token_keep_ratio > 0.75:
        #     reward_4 = -8
        # elif token_keep_ratio <= 0.75 and token_keep_ratio > 0.70:
        #     reward_4 = -4
        # elif token_keep_ratio <= 0.70 and token_keep_ratio > 0.60:
        #     reward_4 = 4 - 2*math.exp(10*abs(token_keep_ratio - 0.65))
        # elif token_keep_ratio <= 0.60 and token_keep_ratio > 0.55:
        #     reward_4 = -4
        # elif token_keep_ratio <= 0.55:
        #     reward_4 = -8
        
        # if token_keep_ratio - 0.65 > 0:
        #     reward_4 = -action*4*(math.exp(token_keep_ratio - 0.65)-1)
        # else:
        #     reward_4 = -(1-action)*4*(math.exp(abs(token_keep_ratio - 0.65))-1)

        # if token_keep_ratio > 0.8:
        #     reward_4 = -action*1*(math.exp(eta*(token_keep_ratio - 0.70))-1)
        # elif token_keep_ratio < 0.6:
        #     reward_4 = -(1-action)*1*(math.exp(eta*abs(token_keep_ratio - 0.70))-1) 
        # else:
        #     # reward_4 = -1*(math.exp((eta)*abs(token_keep_ratio - 0.70))) 
        #     reward_4 = 25*(math.exp((1-token_keep_ratio)) - 1) - 4*math.exp(token_keep_ratio)

        reward_4 = - (math.exp(abs(token_keep_ratio - 0.70))-1)
        # reward_4 = 0


    eta = 0.6
    beta = 0.80
    return eta*reward_1 + (1-eta)*(beta*reward_2 + (1-beta)*reward_4)
    # return reward_2
    # return reward_1 + reward_2

def caculate_reward(num_block, classify_correct, action):
    # size of action: [token_num] -> 197
    # action for 197 tokens in one image

    reward_for_classify = 24 
    # simplest: split equally
    if classify_correct:
        reward_1 = reward_for_classify/12
    else:
        reward_1 = -reward_for_classify/12

    reward_for_action = 1
    reward = torch.empty(action.shape, device=action.device)
    for i in range(len(action)):
        # action: 0:discard token 
        #         1:keep token
        reward_2 = 0
        reward_2 += (1 - action[i])*reward_for_action

        reward_total = reward_1 + reward_2
        reward[i] = reward_total 
        
    return reward


def plot_mask(batch, paths, mask):
    # size of x:[64,3,224,224]
    # size of mask:[3,64,197] 

    images = []
    transform = transforms.Compose([transforms.Resize((224, 224))])
    for path in paths:
        img = Image.open(path)
        if img.mode == 'L':
            img = img.convert("RGB")
        img = transform(img)
        img_array = np.array(img)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        img_tensor = img_tensor.permute(2,0,1)
        images.append(img_tensor)

    x = torch.stack(images)


    # Define your transformations

    _mask = mask[:,:,1:197]
    # size of reshaped_mask:[3,64,14,14]
    reshaped_mask = _mask.view(3,64,14,14)
    # size of repeated_x:[4,64,3,224,224]
    repeated_x = x.repeat(4,1,1,1,1)

    # repeated_x = repeated_x.clamp(0,255) # Ensure pixel values are within [0, 255]
    # repeated_x = repeated_x.to(torch.uint8)  # Convert to unsigned 8-bit integer type
    # for i in range(repeated_x.size(1)):
    #     for j in range(repeated_x.size(0)):
    #         temp = repeated_x[j,i]
    #         img_array= temp.permute(1,2,0).cpu().numpy()
    #         img = Image.fromarray(img_array)
    #         img.save(f'./plot/{batch}_{i}_{j}.png')

    blank_black = torch.zeros(3,224,224)
    blank_white = torch.ones(3,224,224)

    modified_x = repeated_x.clone()
    mask_ = reshaped_mask.clone()
    for i in range(mask_.size(0)):
        for j in range(mask_.size(1)):
            for m in range(mask_.size(2)):
                for n in range(mask_.size(3)):
                    if mask_[i,j,m,n] == 0:
                        modified_x[i+1, j, :, m*16:(m+1)*16, n*16:(n+1)*16] = 255


    # Convert the modified tensor back to images and save them
    modified_x = modified_x.clamp(0,255) # Ensure pixel values are within [0, 255]
    modified_x = modified_x.to(torch.uint8)  # Convert to unsigned 8-bit integer type
    # to_pil = transforms.ToPILImage()
    # size of modified_x:[4,64,3,224,224]
    for i in range(modified_x.size(1)):
        for j in range(modified_x.size(0)):
            temp = modified_x[j,i]
            img_array= temp.permute(1,2,0).cpu().numpy()
            img = Image.fromarray(img_array)
            img.save(f'./plot/{batch}_{i}_{j}.png')
            # img = to_pil(modified_x[j,i])
            # img.save(f'./plot/{batch}_{i}_{j}.png')

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    model.agent_n.eval_()
    # model.train(True)

    for images, targets, paths in metric_logger.log_every(data_loader, 10, header):
    # for images, targets, paths in metric_logger.log_every(data_loader, 10, header):
        # shape of images:[96,3,224,224]
        images = images.to(device, non_blocking=True)
        # shape of target:[96]
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # shape of action_n: [3,96,196]
            output, action_n = model(images)

            loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_(data_loader, model, device, train_batch_num, args):
    global eval_num
    global token_depth_global
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    model.agent_n.eval_()

    keep_ratio = torch.tensor([0,0,0]).to('cuda')
    token_depth  = 0
    batch_num = 0
    for images, targets, paths in metric_logger.log_every(data_loader, 10, header):
    # for images, targets, paths in metric_logger.log_every(data_loader, 10, header):
        # shape of images:[96,3,224,224]
        images = images.to(device, non_blocking=True)
        # shape of target:[96]
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # shape of action_n: [3,96,196]
            output, action_n = model(images)

            action_n_sum = torch.sum(action_n, dim=(1,2))
            keep_ratio_batch = action_n_sum/(96*196)

            keep_ratio = keep_ratio + keep_ratio_batch
            batch_num += 1
            token_depth += 3*196*96
            token_depth += 3*torch.sum(action_n_sum)

            loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # keep_ratio_temp = [x/batch_num for x in keep_ratio]
    # keep_ratio_array = keep_ratio_temp.cpu().numpy()
    keep_ratio = keep_ratio.cpu().numpy()
    token_depth = token_depth.cpu().numpy()

    if train_batch_num == 0:
        eval_num = 0
        token_depth_global = 0
    temp = token_depth/batch_num
    eval_num += 1
    token_depth_global += temp
    token_depth_global_average = token_depth_global/eval_num
    log_stats = {
             **{f'test_{k}': meter.global_avg for k, meter in metric_logger.meters.items()},
             'batch_num': train_batch_num,
             'keep_ratio': [x/batch_num for x in keep_ratio],
             'token_depth': [token_depth/batch_num],
             'token_depth_average': [token_depth_global_average],
            }
    
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



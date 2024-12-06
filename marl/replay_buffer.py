import numpy as np
import torch

class ReplayBuffer():
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode = args.episode
        self.episode_limit = args.episode_limit
        self.episode_num = 0 
        self.total_step = 0
        self.buffer = None
        self.reset_buffer()
        self.index = 0

    def reset_buffer(self):
        self.buffer = {
            'obs_n': torch.empty([self.episode, self.episode_limit, self.N,
                                  self.obs_dim], device='cuda'),
            'v_n': torch.empty([self.episode, self.episode_limit, self.N],device='cuda'),
            'obs_n_': torch.empty([self.episode, self.episode_limit, self.N,
                                   self.obs_dim], device='cuda'),
            'v_n_': torch.empty([self.episode, self.episode_limit, self.N],device='cuda'),
            'a_n': torch.empty([self.episode, self.episode_limit, self.N],device='cuda'),
            'a_logprob_n': torch.empty([self.episode, self.episode_limit, self.N],
                                       device='cuda'),
            'r_n': torch.empty([self.episode, self.episode_limit, self.N],device='cuda'),
            'done_n': torch.empty([self.episode, self.episode_limit, self.N],
                                  device='cuda'),
            'died_win': torch.empty([self.episode, self.episode_limit, self.N],
                                  device='cuda'),
            'done_episode': torch.empty([self.episode, self.episode_limit, self.N],
                                  device='cuda'),
            'cls_token':torch.empty([self.episode, self.episode_limit,
                                     self.obs_dim], device='cuda')
        }
        self.episode_num = 0
        self.index = 0

    # def store_transition(self, episode_step, obs_n, v_n, obs_n_, a_n,
    #                      a_logprob_n, r_n, done_n):
    #     self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
    #     self.buffer['v_n'][self.episode_num][episode_step] = v_n
    #     self.buffer['obs_n_'][self.episode_num][episode_step] = obs_n_
    #     self.buffer['a_n'][self.episode_num][episode_step] = a_n
    #     self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
    #     self.buffer['r_n'][self.episode_num][episode_step] = r_n
    #     self.buffer['done_n'][self.episode_num][episode_step] = done_n
    #     self.total_step += 1

    def store_transition(self,transition): 
        if self.index >=  self.episode:
            self.index = 0
        episode_step = transition.episode_step
        # episode_num = transition.episode_num
        episode_num = self.index
        

        self.buffer['obs_n'][episode_num][episode_step] = transition.obs_n
        self.buffer['v_n'][episode_num][episode_step] = transition.v_n
        self.buffer['obs_n_'][episode_num][episode_step] = transition.obs_n_
        self.buffer['v_n_'][episode_num][episode_step] = transition.v_n_
        self.buffer['a_n'][episode_num][episode_step] = transition.a_n
        self.buffer['a_logprob_n'][episode_num][episode_step] = transition.a_logprob_n
        self.buffer['r_n'][episode_num][episode_step] = transition.r_n
        self.buffer['done_n'][episode_num][episode_step] = transition.done_n
        self.buffer['died_win'][episode_num][episode_step] = transition.died_win
        self.buffer['done_episode'][episode_num][episode_step] = transition.done_episode
        self.buffer['cls_token'][episode_num][episode_step] = transition.cls_token
        self.total_step += 1
        self.index += 1


    # def store_last_value(self, episode_step, v_n):
    #     self.buffer['v_n'][self.episode_num][episode_step]
    #     self.episode_num += 1

    
    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key =='a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch

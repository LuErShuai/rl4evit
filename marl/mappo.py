import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import time
from tensorboardX import SummaryWriter

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        x = self.activate_func(self.fc3(x))
        prob = torch.softmax(self.fc4(x), dim=-1)
        return prob


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        x = self.activate_func(self.fc3(x))
        value = self.fc4(x)
        return value


class MAPPO:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip
        self.rand_die_ratio = args.rand_die_ratio
        timestamp = time.time()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(timestamp))
        self.writer = SummaryWriter('./runs/Agent/loss_{}'.format(formatted_time))
        self.training_step = 0

        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.lr)
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    def choose_action(self, obs_n, evaluate):
        # shape of obs_n: [batch_size, token_num - 1, token_dim]
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                """
                    Add an one-hot vector to represent the agent_id
                    For example, if N=3
                    [obs of agent_1]+[1,0,0]
                    [obs of agent_2]+[0,1,0]
                    [obs of agent_3]+[0,0,1]
                    So, we need to concatenate a N*N unit matrix(torch.eye(N))
                """
                eye = torch.eye((self.N)).to('cuda')
                eye_ = eye.reshape(1, eye.shape[0], eye.shape[1])
                eye_batch = eye_.repeat([obs_n.shape[0], 1, 1])
                actor_inputs.append(eye_batch)

            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_input.shape=(N, actor_input_dim)
            prob = self.actor(actor_inputs)  # prob.shape=(N,action_dim)
            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                a_n = prob.argmax(dim=-1)
                return a_n.numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                # return a_n.numpy(), a_logprob_n.numpy()
                if self.rand_die_ratio != 1:
                    mask = np.random.choice([0,1],size=a_n.shape,p=[self.rand_die_ratio,1-self.rand_die_ratio])
                    mask_tensor = torch.tensor(mask, device=a_n.device)
                    a_n = mask_tensor
                return a_n, a_logprob_n

    def get_value(self, s):
        # shape of input s : [64, 196, 768]
        # shape of output v_n : [64, 196]
        # we need to construct a global feature as the input of critic
        # for each agent, the global feature is concat(token, cls_token)
        # state_n.shape: [batch_size, token_num, token_dim]:[64,197,768]
        # cls_token.shape: [batch_size, 1, token_dim]
        # batch_size here means the batch size in dvit
        with torch.no_grad():

            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            # s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            state_n = s[:,1:197,:]
            cls_token = s[:,0,:].reshape(state_n.shape[0], 1, state_n.shape[2])
            # state_n = obs_n
            # cls_token = cls_token.reshape(state_n.shape[0], 1, state_n.shape[2])
            cls_token_n = cls_token.repeat([1, state_n.shape[1],1])
            
            state_global = torch.cat((state_n, cls_token_n), axis=-1)

            critic_inputs.append(state_global)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                eye = torch.eye((self.N)).to('cuda')
                eye_ = eye.reshape(1, eye.shape[0], eye.shape[1])
                eye_batch = eye_.repeat([state_n.shape[0], 1, 1])
                critic_inputs.append(eye_batch)
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
            # cls_token_global = np.tile(cls_token, (cls_token.shape[0],
            #                                        state_n.shape[1], cls_token[2]))
            # critic_inputs = np.concatenate((state_n, cls_token_global), axis=-1)
            v_n = self.critic(critic_inputs)  # v_n.shape(B,N,1)
            # return v_n.numpy().flatten()
            return v_n.squeeze()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # get training data
        
        # done_n = 1 means dead
        # done_episode = 1 means ends of episode, conclude dead, win
        # or reach the max episode_step
        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            # deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]  # deltas.shape=(batch_size,episode_limit,N)
            deltas = batch['r_n'] + self.gamma * batch['v_n_'] * (1 - batch['done_n']) - batch['v_n']  # deltas.shape=(batch_size,episode_limit,N)
            # deltas = batch['r_n'] + self.gamma * batch['v_n_'] * (1 - batch['died_win']) - batch['v_n']  # deltas.shape=(batch_size,episode_limit,N)
            # deltas = batch['r_n'] + self.gamma * batch['v_n_'] - batch['v_n']  # deltas.shape=(batch_size,episode_limit,N)
            # for t in reversed(range(self.episode_limit)):
            #     temp = batch['done_n']
            #     gae = deltas[:, t] + self.gamma * self.lamda * gae *(1-batch['done_episode'][:, t])
            #     # gae = deltas[:, t] + self.gamma * self.lamda * gae * (1-batch['done_episode'][:, t])
            #     # gae = deltas[:, t] + self.gamma * self.lamda * gae
            #     adv.insert(0, gae)
            # adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            # # v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
            # v_target = adv + batch['v_n']  # v_target.shape(batch_size,episode_limit,N)
            # if self.use_adv_norm:  # Trick 1: advantage normalization
            #     adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

            adv = deltas.detach()
            # if self.use_adv_norm:  # Trick 1: advantage normalization
            #     adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
            v_target = adv + batch['v_n']

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        # [64,3,196,768] [64,3,196,1536]
        actor_inputs, critic_inputs = self.get_inputs(batch)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []
                    for t in range(self.episode_limit):
                        prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1)) # prob.shape=(mini_batch_size*N, action_dim)
                        probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))  # prob.shape=(mini_batch_size,N,action_dim）
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))  # v.shape=(mini_batch_size*N,1)
                        values_now.append(v.reshape(self.mini_batch_size, self.N))  # v.shape=(mini_batch_size,N)
                    # Stack them according to the time (dim=1)
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    probs_now = self.actor(actor_inputs[index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, episode_limit, N)
                # batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])  # a_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())  # ratios.shape=(mini_batch_size, episode_limit, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2)
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                if self.use_value_clip:
                    # values_old = batch["v_n"][index, :-1].detach()
                    values_old = batch["v_n_"][index, :].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2
                    max_0 = torch.max(critic_loss)
                    min_0 = torch.min(critic_loss)
                    max_1 = torch.max(values_now)
                    min_1 = torch.min(values_now)
                    max_2 = torch.max(v_target[index])
                    min_2 = torch.min(v_target[index])

                self.ac_optimizer.zero_grad()
                a = actor_loss.mean()
                b = torch.mean(actor_loss)
                c = critic_loss.mean()
                d = torch.mean(critic_loss)
                max_d = torch.max(critic_loss)
                min_d = torch.min(critic_loss)
                sum_d = torch.sum(critic_loss)

                # ac_loss = actor_loss.mean() + critic_loss.mean()
                # ac_loss.backward()
                # if self.use_grad_clip:  # Trick 7: Gradient clip
                #     torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                # self.ac_optimizer.step()

                # for param in self.actor.parameters():
                #     if param.grad is None:
                #         print(f"Gradient for  actor parameter is 'None'.")
                #         print(self.training_step)
                # for param in self.critic.parameters():
                #     if param.grad is None:
                #         print(f"Gradient for critic parameter is 'None'.")
                #         print(self.training_step)
                        # print(f"Gradient for parameter {name} is not 'None'.")

                self.actor_optimizer.zero_grad()
                actor_loss.mean().backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.mean().backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.critic_optimizer.step()
                max_grad_value_1 = torch.max(torch.cat([param.grad.view(-1) for param in self.actor.parameters()]))
                max_grad_value_2 = torch.max(torch.cat([param.grad.view(-1) for param in self.critic.parameters()]))

                self.writer.add_scalar('loss/actor_grad', max_grad_value_1, global_step=self.training_step)
                self.writer.add_scalar('loss/critic_grad', max_grad_value_2, global_step=self.training_step)
                self.writer.add_scalar('loss/action_loss', actor_loss.mean(), global_step=self.training_step)
                self.writer.add_scalar('loss/critic_loss', critic_loss.mean(), global_step=self.training_step)
                self.training_step += 1

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        self.writer.add_scalar('lr', lr_now, global_step=total_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        # shape of obs_n: [batch_size, episode_step, token_num, token_dim]
        # shape of cls_token: [batch_size, episode_step, token_dim]
        obs_n = batch['obs_n']
        cls_token = batch['cls_token']
        cls_token = cls_token.reshape(cls_token.shape[0],cls_token.shape[1],1,cls_token.shape[2])
        cls_token_n = cls_token.repeat([1,1,obs_n.shape[2],1])

        state_global = torch.cat((obs_n, cls_token_n), axis=-1)

        actor_inputs.append(obs_n)
        critic_inputs.append(state_global)
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N,device='cuda').unsqueeze(0).unsqueeze(0).repeat(obs_n.shape[0], obs_n.shape[1], 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), "./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load("./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))

    def eval_(self):
        self.actor.eval()
        self.critic.eval()

    def save_agent_n(self):
        torch.save(self.actor.state_dict(), "./param/MAPPO_ACTOR.pkl")
        torch.save(self.critic.state_dict(), "./param/MAPPO_CRITIC.pkl")

    def save_agent_n_(self, path):

        _path_actor = "./param/" + str(path) + "_actor.pkl"
        _path_critic = "./param/" + str(path) + "_critic.pkl"
        torch.save(self.actor.state_dict(), _path_actor)
        torch.save(self.critic.state_dict(), _path_critic)

    def load_agent_n_actor(self):
        state_dict_actor = torch.load("./param/MAPPO_ACTOR.pkl")
        self.actor.load_state_dict(state_dict_actor)

    def load_agent_n_critic(self):
        state_dict_critic = torch.load("./param/MAPPO_CRITIC.pkl")
        self.critic.load_state_dict(state_dict_critic)

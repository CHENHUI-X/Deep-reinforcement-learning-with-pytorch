import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# from tensorboardX import SummaryWriter

# Parameters
gamma = 0.99
render = True
seed = 1
log_interval = 10

env = gym.make('CartPole-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 100
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        # self.actor_net = Actor()
        # self.critic_net = Critic()
        self.actor_net = Actor().float()
        self.critic_net = Critic().float()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        # self.writer = SummaryWriter('../exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        # if not os.path.exists('../param'):
        #     os.makedirs('../param/net_param')
        #     os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()


    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1)
        next_state = torch.tensor(np.array([t.next_state for t in self.buffer]), dtype=torch.float)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1)
        # old 对应 theta' 这一系列 st at rt 都是基于theta'得到的,接下来用这些st去不断更新theta

        '''
        # R = 0
        # Gt = []
        # for r in reward[::-1]:
        #     R = r + gamma * R
        #     Gt.insert(0, R)
        #     # 保证最开始的action得到的reward在最前边
        #     # 那么 Gt 中每一个元素 就是存储 该节点 所采取相应行动后,得到的累计奖励
        # Gt = torch.tensor(Gt, dtype=torch.float)
        #
        # 这里的写法是把 reward 写成rt + gama * rt+1 + gama^2 * rt+2 ...
        # 然后用reward - critic(st) = advantage 表示st采取at后,相比原来好了多少
        # 而TD_PPO2中,是直接用 rt + gama * critic(next_state) ,即critic(next_state)
        # 给出的是一个,在采取at后,st+1后reward的一个期望,
        # 代替了后续 rt+1 , rt+2 ...具体操作带来的reward
        # 这里,我注释掉了,本来的写法,还是采取的TD_PPO2中的写法
        '''
        with torch.no_grad():
            target_v = reward + gamma * self.critic_net(next_state)
            #  target_v 指的是,在St画面,具体采取了at这个动作,到最后得到的reward
        advantage = (target_v - self.critic_net(state)).detach()  # Advantage actor-critic
        # 这是advantage就是指,具体st采取at,相比st平均采取所有动作,得到的分数,多了多少
        # 在 旧的theta上得到的 A_theta' 结果

        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(
                  SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False
            ):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                '''
                # with torch.no_grad():
                # Gt_index = Gt[index].view(-1, 1)
                #
                # V = self.critic_net(state[index]) #
                #
                # delta = Gt_index - V
                # # Gt_index - V  : V 理解为出现St时,平均得到的分数,Gt是采取a(t)后最后得到的分数
                # advantage = delta.detach()
                
                # 这里的写法是把 reward 写成rt + gama * rt+1 + gama^2 * rt+2 ...
                # 然后用reward - critic(st) = advantage 表示st采取at后,相比原来好了多少
                # 而TD_PPO2中,是直接用 rt + gama * critic(next_state) ,即critic(next_state)
                # 给出的是一个,在采取at后,st+1后reward的一个期望,
                # 代替了后续 rt+1 , rt+2 ...具体操作带来的reward
                # 这里,我注释掉了,本来的写法,还是采取的TD_PPO2中的写法
                
                '''

                # epoch iteration, PPO core!!!
                # action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy
                # 每次新的p(theta tao)
                action_log_prob = self.actor_net(state[index])


                ratio = (action_log_prob / old_action_log_prob[index])
                surr1 = ratio * advantage[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])

                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience


def main():
    agent = PPO()
    for i_epoch in range(1000):
        state = env.reset()

        for t in count():
            action, action_prob = agent.select_action(state)
            next_state, reward,  done ,truncated, _  = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            if  render  :  env.render()
            agent.store_transition(trans)
            state = next_state

            if done: break

        if  agent.counter % agent.buffer_capacity:
            agent.update(i_epoch)


if __name__ == '__main__':
    main()
    print("end")
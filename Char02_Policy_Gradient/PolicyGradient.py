
import argparse
import gymnasium as gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

'''
Maybe this file should name "naive policy gradient"

'''

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')

env.action_space.seed(args.seed)

torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2) # 0 for left , 1 for right

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state) # prob

    m = Categorical(probs) # create a category distribution according pro
    action = m.sample() # sample an action from distribution ,output 0 or 1
    policy.saved_log_probs.append(m.log_prob(action)) # 推导中的log p(at|st)
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        # gama^n * r1 + gama^(n-1) * r2 + ... + gama^0 * rn
        rewards.insert(0, R)
        # 保证最开始的action得到的reward在最前边
        # 那么rewards中每一个元素 就是存储 该节点 所采取相应行动后,得到的累计奖励

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward() # -min <=> +max , 所以 -log_prob
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1): # 无限循环
        state , info = env.reset()
        for t in range(100):
            # Don't infinite loop while learning
            action = select_action(state)
            next_state, reward,  done ,truncated, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            # 将当前episode的每一个action的reward都记录
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01

        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()

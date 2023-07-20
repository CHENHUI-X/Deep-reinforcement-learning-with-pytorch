import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import copy

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

'''
#  unwrapped  还原env的原始设置，env外包了一层防作弊层 
据说gym的多数环境都用TimeLimit（源码）包装了，以限制Epoch，
就是step的次数限制，比如限定为200次。所以小车保持平衡200步后，就会失败
env._max_episode_steps : 200
-------------------------------------------------------------
用env.unwrapped可以得到原始的类，原始类想step多久就多久，不会200步后失败：
env.unwrapped : gym.envs.classic_control.cartpole.CartPoleEnv
'''

env = gym.make("CartPole-v0").unwrapped
state, info = env.reset()

NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = env.action_space.shape

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        # indicate that Q value of all  action  with state s

        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO: # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy() #
            # max return value and index ,the index is the action
            action = action[0]
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY # 2000个index 轮流覆盖
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # update target net
        self.learn_step_counter += 1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        '''
                # torch.gather(input, dim, index)
                b = torch.Tensor([[1,2,3],[4,5,6]])
                print b
                index_1 = torch.LongTensor([[0,1],[2,0]])
                index_2 = torch.LongTensor([[0,1,1],[0,0,0]])
                print torch.gather(b, dim=1, index=index_1)
                print torch.gather(b, dim=0, index=index_2)

                输出 : 

                     1  2  3
                     4  5  6
                    [torch.FloatTensor of size 2x3]

                     1  2
                     6  4
                    [torch.FloatTensor of size 2x2]
                    就是把索引替换为index,只不过看替换的是哪个维度,dim=1,index = [[0,1],[2,0]]
                    那调用的是 (0,0) (0,1)  (1,2) (1,0)
                    结果分别为   1      2     6     4

                     1  5  6
                     1  2  3
                    [torch.FloatTensor of size 2x3]
                    当dim = 0,index = [[0,1,1],[0,0,0]]
                    调用的是 (0,0) (1,1) ( 1,2)  (0,0) ( 0,1) (0,2)
                    结果分别为  1     5     6       1      2     3

                '''
        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        # get action use [ batch , action_index]

        q_next = self.target_net(batch_next_state).detach() # Q(st+1,a)
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward
def plot_live( reward_list ) :
    plt.ion()
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(reward_list)

    plt.pause(0.0000001)

def main():
    dqn = DQN()
    episodes = 400
    print("Collecting Experience....")
    reward_list = []

    for i in range(episodes):
        state, info = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state)

            next_state, reward,  done , truncated , _  = env.step(action)
            position, velocity ,pos_angle, v_angle = next_state
            reward = reward_func(env, position, velocity ,pos_angle, v_angle)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                """
                只有exp池中transition 数量足够多之后,才会开始进行学习
                
                """
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
                # else go to next state
            state = next_state
        r = copy.copy(ep_reward)
        reward_list.append(r)
        # plot
        plot_live(reward_list)


if __name__ == '__main__':
    main()

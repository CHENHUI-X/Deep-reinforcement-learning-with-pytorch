# https://blog.paperspace.com/getting-started-with-openai-gym/
import gym
import matplotlib.pyplot as plt
import time

print(gym.version.VERSION)

env = gym.make("BreakoutNoFrameskip-v4",
               render_mode='human')
#
# print("Observation Space: ", env.observation_space)
# print("Action Space       ", env.action_space)
# obs = env.reset()
#
# for i in range(1000):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     time.sleep(0.01)
# env.close()

'''
Our observation space is a continuous space of dimensions (210, 160, 3) 
corresponding to an RGB pixel observation of the same size. 
Our action space  contains 4 discrete actions (Left, Right, Do Nothing, Fire)
------------------------------------------------------------------------------

It's a common practice in Deep RL that we construct our observation by 
concatenating the past  *** k frames ***  together.

------------------------------------------------------------------------------
For this we define a class of type gym.Wrapper to override the reset 
and return functions of the Breakout Env.

------------------------------------------------------------------------------
We modify the observation space from (210, 160, 3) to (210, 160, 3 * num_past_frames.

'''

from collections import deque
from gym import spaces
import numpy as np

class ConcatObs(gym.Wrapper):
    # 要对gym 重新包装
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k # k frames
        self.frames = deque([], maxlen=k)
        # https://blog.csdn.net/weixin_43790276/article/details/107749745

        shp = env.observation_space.shape
        self.observation_space = \
            spaces.Box(
                low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype
            )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob) # 初始化reset的时候,用的是初始帧,只不过重复了4次
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        # 初始的时候,里边有k张初始帧,每次更新一个步骤,输出一个新的state:ob
        # 然后将ob追加到队列,因为设置了max-length = k , 所以队列会自动删除最旧一个
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.array(self.frames)

env = gym.make("BreakoutNoFrameskip-v4",
               render_mode='human')
wrapped_env = ConcatObs(env, 4) # 包装以后的env
print("The new observation space is", wrapped_env.observation_space)

# Reset the Env
obs = wrapped_env.reset()
print("Intial obs is of the shape", obs.shape)

# Take one step
obs, _, _, _  = wrapped_env.step(2)
print("Obs after taking a step is", obs.shape)

'''
There is more to Wrappers than the vanilla Wrapper class.
Gym also provides you with specific wrappers that target 
specific elements of the environment, such as observations, 
rewards, and actions. Their use is demonstrated in the following section.

-----------------------------------------------------------------------
ObservationWrapper: This helps us make changes to the observation using 
                        the observation method of the wrapper class.
                        
RewardWrapper: This helps us make changes to the reward using the 
                        reward function of the wrapper class.
                        
ActionWrapper: This helps us make changes to the action using the 
                        action function of the wrapper class.
-------------------------------------------------------------------------
Let us suppose that we have to make the follow changes to our environment:
        We have to normalize the pixel observations by 255.
        We have to clip the rewards between 0 and 1.
        We have to prevent the slider from moving to the left (action 3).

'''
import random

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # Normalise observation by 255
        return obs / 255.0


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # Clip reward between 0 to 1
        return np.clip(reward, 0, 1)


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        if action == 3:
            return random.choice([0, 1, 2])
        else:
            return action

wrapped_env = ObservationWrapper(
    RewardWrapper(
        ActionWrapper(
            wrapped_env
        )
    )
) # wher wrapped_env = ConcatObs(env, 4) # 包装以后的env

obs = wrapped_env.reset()

for step in range(500):
    action = wrapped_env.action_space.sample()
    obs, reward, done, info = wrapped_env.step(action)
    print(obs.shape )
    # Raise a flag if values have not been vectorised properly
    if (obs > 1.0).any() or (obs < 0.0).any():
        print("Max and min value of observations out of range")

    # Raise a flag if reward has not been clipped.
    if reward < 0.0 or reward > 1.0:
        assert False, "Reward out of bounds"


    time.sleep(0.001)

wrapped_env.close()

print("All checks passed")
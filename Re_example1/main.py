

# https://blog.paperspace.com/getting-started-with-openai-gym/
# https://brosa.ca/blog/ale-release-v0.7

import gym

import time
print(gym.version.VERSION)
env = gym.make('MountainCar-v0')

# Observation and action space
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

'''
the action space like this  :
The Discrete(n) box describes a discrete space with [0.....n-1]
possible values. In our case n = 3, meaning our actions can
take values of either 0, 1, or 2.

There are multiple other spaces available for various use cases,
such as MultiDiscrete, which allow you to use more than
one discrete variable for your observation and action space.

'''

import matplotlib.pyplot as plt

# reset the environment and see the initial observation
obs = env.reset()
print("The initial observation is {}".format(obs)) # (velocity , position)

# Sample a random action from the entire action space
random_action = env.action_space.sample()

# # Take the action and get the new observation space
new_obs, reward, done, info = env.step(random_action)
print("The new observation is {}".format(new_obs)) # new v and pos

'''
In this case, our observation is not the screenshot of the task
being performed. In many other environments (like Atari, as we will see),
the observation is a screenshot of the game. In either of the scenarios,
if you want to see how the environment looks in the current state,
you can use the render method.

'''
env.render(mode = "human") # the state of car or observation
'''
If you want to see a screenshot of the game as an image,
rather than as a pop-up window, you should set the mode
argument of the render function to rgb_array

'''
env_screen = env.render(mode = 'rgb_array') # get a frame from observation

env.close()

plt.imshow(env_screen) # show the screenshot

'''
Collecting all the little blocks of code we have covered so far,
the typical code for running your agent inside the MountainCar
environment would look like the following. In our case we just
take random actions, but you can have an agent that does something
more intelligent based on the observation you get.

'''

import time
env = gym.make('MountainCar-v0',
               render_mode='human')

# Number of steps you run the agent for
num_steps = 100

obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs)
    action = env.action_space.sample()

    # apply the action
    obs, reward, done, info = env.step(action)

    # Render the env
    # env.render() # removed

    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.001)

    # If the epsiode is up, then start another one
    if done:
        env.reset()

# Close the env
env.close()

print("Upper Bound for Env Observation",
                env.observation_space.high)
print("Lower Bound for Env Observation",
                env.observation_space.low)


import gym
import time
env = gym.make("Taxi-v3")
print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)

obs = env.reset()
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    env.render()
    time.sleep(0.01)
env.close()

action = env.action_space.sample()
# The action 0-5 corresponds to the actions
# (south, north, east, west, pickup, dropoff)
obs, reward, done, info = env.step(action)
print(action,obs, reward, done, info)
pos = list(env.decode(obs)) # encode(1,1,1,1) -> 234

print(
    f'当前位置 : taxi row -> {pos[0]} , taxi column -> {pos[1]} , passenger index -> {pos[2]}, destination index -> {pos[3]}'
)

# ----------------reward table--------------
'''
When the Taxi environment is created, there is an initial Reward table that's also created,
called `P`. We can think of it like a matrix that has the number of states as rows
and number of actions as columns, i.e. a  shape with (states , actions) matrix

env.P[328] 
Out[133]: 
{0: [(1.0, 428, -1, False)],
 1: [(1.0, 228, -1, False)],
 2: [(1.0, 348, -1, False)],
 3: [(1.0, 328, -1, False)],
 4: [(1.0, 328, -10, False)],
 5: [(1.0, 328, -10, False)]}


{action: [(probability, nextstate, reward, done)]}
- In this env, probability is always 1.0.
- The nextstate is the state we would be in if 
    we take the action at this index of the dict
    

'''

# ---------------------使用穷举法试试--------------
env.s = 328  # set environment to illustration's state

epochs = 0
penalties, reward = 0, 0

frames = []  # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
    }
    )

    epochs += 1

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
print_frames(frames)


# ------------------------Wrappers------------------------
'''
The Wrapper class in OpenAI Gym provides you with the functionality 
to modify various parts of an environment to suit your needs.

But before we begin, let's switch to a more complex environment that 
will really help us appreciate the utility that Wrapper brings to 
the table.
'''

# # -------------- for v5 , that's the latest version , official recommend
# env = gym.make(
#     'ALE/Breakout-v5',
#     obs_type='rgb',                   # ram | rgb | grayscale
#     frameskip=4,                      # frame skip
#     mode=None,                        # game mode, see Machado et al. 2018
#     difficulty=None,                  # game difficulty, see Machado et al. 2018
#     repeat_action_probability=0.25,   # Sticky action probability
#     full_action_space=False,          # Use all actions
#     render_mode='human'                  # None | human | rgb_array
# )

# but here has a error :
# FileNotFoundError: Could not find module
# 'E:\Work_software\python\lib\site-packages\atari_py\ale_interface\ale_c.dll'
# (or one of its dependencies). Try using the full path with constructor syntax.
# ---------------------------------------------------------------------------------

# # -------------- for v0 and v4 ,the old version
# they are legacy but it looks worked
env = gym.make("SpaceInvaders-v4")
print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)

obs = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)
env.close()























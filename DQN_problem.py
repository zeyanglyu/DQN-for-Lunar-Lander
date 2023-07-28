# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import MyAgent
from collections import deque, namedtuple

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """

    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)


def running_average(x, N):
    """ Function used to compute the running average
        of the last N elements of a vector x
    """
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


EPSILON_MAX = 0.99
EPSILON_MIN = 0.05
class EpsilonSample:
    def __init__(self, Z, n_actions):
        self.Z = Z  # Z=0.9 * N_episodes
        self.n_actions = n_actions

    def select(self, k, values):
        # compute current epsilon
        e_k = max(EPSILON_MIN, EPSILON_MAX - (EPSILON_MAX - EPSILON_MIN) * k / (self.Z - 1))

        # roll dice

        if np.random.uniform() > e_k:
            return values.max(1)[1].item()

        else:
            return np.random.randint(0, self.n_actions)
# Import and initialize the discrete Lunar Lander Environment


env = gym.make('LunarLander-v2')
env.reset()

### Create Experience replay buffer ###
L = 15000
buffer = ExperienceReplayBuffer(maximum_length=L)

# Parameters
N_episodes = 550  # Number of episodes
discount_factor = 0.985  # Value of the discount factor
n_ep_running_average = 50  # Running average of 50 episodes
n_actions = env.action_space.n  # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
N = 24  # BATCH SIZE
# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []  # this list contains the total reward per episode
episode_number_of_steps = []  # this list contains the number of steps per episode
loss_list = []
greedy_sampler = EpsilonSample(int(0.9 * N_episodes), n_actions)
# Agent initialization
agent = MyAgent(n_actions)

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
print('N_episodes: {}\ndiscount_factor: {}\nL: {}\nN: {}\n'.format(N_episodes, discount_factor, L, N))
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
  for k in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    epsilon = max(EPSILON_MIN, EPSILON_MAX - (k / N_episodes) * (EPSILON_MAX - EPSILON_MIN))

    while not done:
        env.render()
        if t % int(L / N) == 0:
            agent.update_target_network()
        # Take a random action
        state_tensor = torch.tensor([state],
                                    requires_grad=True,
                                    dtype=torch.float32)

        values = agent.forward(state, epsilon, state_tensor)
        action = greedy_sampler.select(k, values)
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Append experience to the buffer
        exp = Experience(state, action, reward, next_state, done)
        buffer.append(exp)

        # Update episode reward
        total_episode_reward += reward

        ### TRAINING ###
        # Perform training

        if len(buffer) >= N:
            # Sample a batch of N elements
            agent.backward(*buffer.sample_batch(n=N), discount_factor)

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            k, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))

# Saves the agent network
torch.save(agent.network,
           'neural-network-1' + str(running_average(episode_reward_list, n_ep_running_average)[-1]) + '.pth')

# Plot Rewards and steps
_, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes + 1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()

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
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle


class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray, network, epsilon):
        ''' Performs a forward computation '''
        pass

    def backward(self, state: np.ndarray):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


class MyNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, 50, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(50, output_size, dtype=torch.float32)
        self.input_layer_activation = torch.nn.ReLU()

    def forward(self, x):
        out_l1 = self.input_layer(x)
        out_l1_a = self.input_layer_activation(out_l1)
        NetOutput = self.output_layer(out_l1_a)
        return NetOutput


class MyAgent(Agent):

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None
        self.network = MyNet(8, n_actions)
        self.target_network = pickle.loads(pickle.dumps(self.network))
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.002)

    def update_target_network(self):
        self.target_network = pickle.loads(pickle.dumps(self.network))

    def forward(self, state: np.ndarray, epsilon, state_tensor):
        ''' Performs a forward computation '''
        values = self.network.forward(state_tensor)
        return values

    def backward(self, states, actions, rewards, next_states, dones, discount_factor):
        # Training process, set gradients to 0
        self.optimizer.zero_grad()
        # Compute output of the network given the states batch
        values = self.network.forward(torch.tensor(states,
                                                      requires_grad=True,
                                                      dtype=torch.float32))

        # Computes the target values for the states batch
        target_values = self.target_network.forward(torch.tensor(next_states,
                                                                    requires_grad=True,
                                                                    dtype=torch.float32))
        Q_target = list()

        Q_values = torch.zeros(1, len(dones),
                               requires_grad=False,
                               dtype=torch.float32)
        i = 0
        for done in dones:
            if done is True:
                Q_target.append(rewards[i])
            else:
                Q_target.append(rewards[i] + discount_factor * target_values[i].max(0)[0].item())
            Q_values[0, i] = values[i, actions[i]]
            i += 1

        Q_target = torch.tensor([Q_target],
                                requires_grad=True,
                                dtype=torch.float32)

        ''' Performs a backward pass on the network '''
        # Compute loss function
        loss = nn.functional.mse_loss(Q_values, Q_target)

 # Compute gradient

        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.)

        # Perform backward pass (backpropagation)
        self.optimizer.step()


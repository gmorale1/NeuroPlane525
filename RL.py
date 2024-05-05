import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import optim
import copy
import random
import torch
import torch.nn as nn
from DQN_agent import DQN_agent

class Plane_rl:
    def __init__(self, dims):
        self.net = DQN_agent(dims)
        self.init = self.net.init_weights
        self.net.apply(self.init)
        # criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.target_net = copy.deepcopy(self.net)
        self.count = 0
        self.gamma = 0.95
    

    def train(self, pattern_set):
        """
        Update Q-values using pattern set
        """
        # (State, action) and respective target Q-values in a batch
        # random.shuffle(pattern_set)
        # state_action_b, target_q_values = pattern_set
        epoch = 10
        loss_collection = np.zeros(epoch)

        # for i in range(self.args.agent_epochs):
        for i in range(epoch):
            
            random.shuffle(pattern_set)
            state_action_b, target_q_values = zip(*pattern_set[:100])
            # state_action_b, target_q_values = zip(*pattern_set)
            state_action_b = torch.stack(state_action_b)
            target_q_values = torch.stack(target_q_values)

            predicted_q_values = self.net(state_action_b).squeeze()

            predicted_q_values_1 = predicted_q_values[:, 0]  # Assuming the first Q-value is for action 1
            predicted_q_values_2 = predicted_q_values[:, 1]  # Assuming the second Q-value is for action 2

            # Compute the MSE loss for each Q-value separately
            loss_1 = nn.functional.mse_loss(predicted_q_values_1, target_q_values[:, 0])
            loss_2 = nn.functional.mse_loss(predicted_q_values_2, target_q_values[:, 1])

            # Compute the total loss as the sum of the losses for each Q-value
            loss = loss_1 + loss_2

            # loss = nn.functional.mse_loss(predicted_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_collection[i] = loss.item()

        return np.array(loss_collection), loss.item()
    
    def get_action_with_probability(self, r, remaining):
        """
        Helper function to get action based on probability.
        Uniformly distribute them into 0 and 1
        """
        half = remaining / 2
        return 0 if r < half else 1

    def exponential_ep_greedy(self, ep=None, episodes= 40):

        r1 = random.random()
        r2 = random.random()
        # Exponential decay percentage
        remaining = np.exp(-0.015 * ep)
        throttle = -1
        angle = -1

        if r1 < remaining:
            # Take random action
            throttle = self.get_action_with_probability(r1, remaining)

        if r2 < remaining:
            # Take random action
            angle = self.get_action_with_probability(r1, remaining)

        return throttle, angle
    
    def get_action(self, ep, state):
        throttle, angle = self.exponential_ep_greedy(ep)
        if throttle == -1 and angle == -1:
            throttle, angle = self.net(torch.FloatTensor(state))
        return throttle, angle
    
    def generate_pattern_set(self, experiences):
        """
        Pattern set = supervised dataset from transitions
        """

        # added here
        self.count += 1
        if self.count % 2 == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            self.count = 0
        print(f"Count: {self.count}")

        # random.shuffle(experiences)
        states, actions, scores, next_states, dones = zip(*experiences)
        
        # b means batch
        state_b = torch.FloatTensor(np.array(states))
        # action_b = torch.FloatTensor(actions)
        cost_b = torch.FloatTensor(scores)
        next_state_b = torch.FloatTensor(np.array(next_states))
        done_b = torch.FloatTensor(dones)

        # state_action_b = torch.cat([state_b, action_b.unsqueeze(1)], 1)
        # assert state_action_b.shape == (len(experiences), state_b.shape[1] + 1)


        # outputs = self.target_net(torch([next_state_b], 1)).squeeze()
        outputs = self.target_net(torch.tensor(next_state_b, dtype=torch.float32)).squeeze()
        # get score from outputs which would be our q_next_state_b
        with torch.no_grad(): # TODO: remove this no grad?
            target_q_values_1 = cost_b + self.gamma * outputs[:, 0] * (1 - done_b)
            target_q_values_2 = cost_b + self.gamma * outputs[:, 1] * (1 - done_b)
            target_q_values = (target_q_values_1, target_q_values_2)


        # Return the supervised dataset
        return (state_b, target_q_values)

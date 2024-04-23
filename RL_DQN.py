"""
Among other things, the cost function and other setup necessary to throttle
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optim
import copy
import random

class DQN:
    def __init__(self, args, env, inputs):
        self.env = env
        self.args = args 
        self.net = NetWithoutdropout(self.args.num_params) 
        self.optimizer = optim.Adam(self.net.parameters()) # Rprop is the default for NFQ
        self.target_net = copy.deepcopy(self.net)
        self.count = 0
        self.train_episodes = 30

        self.step_cost = 0.001
    


    def reset(self):

        return state

    def step(self, action):
        """
        Explain the reward function here
        """
        
        state = self.env.step(action)                
        #--------#
        return state, score, failed

    def close(self):
        self.env.close()

    def get_action_with_probability(self, r, remaining):
        """
        Helper function to get action based on probability.
        Uniformly distribute them into 0 and 1
        """
        half = remaining / 2
        return 0 if r < half else 1

    def exponential_ep_greedy(self, ep=None, episodes= None):

        r = random.random()
        # Exponential decay percentage
        remaining = np.exp(-0.015 * ep)

        if r < remaining:
            # Take random action
            return self.get_action_with_probability(r, remaining)
        else:
            return -1

    def main(self,):
        all_experiences = []
        for ep in range(1, self.train_episodes+1): 
            print(f"Train Episode: {ep}")

            # Which strategy to use for exploration
            exploration = self.exponential_ep_greedy()

            # Perform an agent rollout
            # During training, emergency braking has to be checked at every time-step while gathering experience
            success, new_experiences, episode_score = self.experience(
                self.net,
                exploration,
                ep,
                self.args.train_episodes,
                self.args.train_max_steps)
            
            all_experiences.extend(new_experiences)
            state_action_b, target_q_values = self.generate_pattern_set(all_experiences)

            combined_data = list(zip(state_action_b, target_q_values))
            
            loss_colection, _ = self.train(combined_data)
            avg_loss_episode = np.mean(loss_colection)

    def get_best_action(self, state):
        """
        Make copies of the network and evaluate Q-value for each (state, action) combination
        Our controller is Bang-bang, can apply either 0V or Full voltage
        """
        # state, action= 0
        outputs = self.net(torch.cat([torch.FloatTensor(state), torch.FloatTensor([0])], dim=0))
        
        # ...
        # If more actions are present (e.g., PWM, first modify the ouput of the NN and add more action Q's here)
        # ...

        # Lower Q value is better, return that
        return outputs

    def experience(self, nfq_agent, action_function, episode_num, total_episodes, max_steps):
        """
        get_action_function is the exploration strategy function, easy to get confused here
        """

        state = self.reset()
        experiences = []

        total_cost = float(0.0)
        success_indicator = 0
        emergency_brake = False

        for step in range(max_steps):

            # Exploration vs Exploitation
            random_action = action_function(episode_num, total_episodes)
            if random_action == -1:
                if self.args.verbose == 1:
                    print("\tExploitation, Random action NOT taken")
                # If action is -1, get the best action based on the current state
                action = nfq_agent.get_best_action(state) 
            else: 
                if self.args.verbose == 1:
                    print("\tExploration, Random action taken")
                action = random_action

            next_state, score, failed = self.step(action)
            # print('\t')
            print(next_state)

            total_score += float(score)
            experiences.append((state, action, score, next_state, failed)) # Add the tuple

            # Add a emergency stop during training (Kill switch)

        return success_indicator, experiences, total_score
    
    def generate_pattern_set(self, experiences):
        """
        Pattern set = supervised dataset from transitions
        """

        # added here
        self.count += 1
        if self.count % 10 == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            self.count = 0
        print(f"Count: {self.count}")

        # random.shuffle(experiences)
        states, actions, scores, next_states, dones = zip(*experiences)
        
        # b means batch
        state_b = torch.FloatTensor(np.array(states))
        action_b = torch.FloatTensor(actions)
        cost_b = torch.FloatTensor(scores)
        next_state_b = torch.FloatTensor(np.array(next_states))
        done_b = torch.FloatTensor(dones)

        state_action_b = torch.cat([state_b, action_b.unsqueeze(1)], 1)
        assert state_action_b.shape == (len(experiences), state_b.shape[1] + 1)


        outputs = self.target_net(torch([next_state_b], 1)).squeeze()
        # get score from outputs which would be our q_next_state_b
        q_next_state_b = .....

        with torch.no_grad(): # TODO: remove this no grad?
            target_q_values = cost_b + self.args.gamma * q_next_state_b * (1 - done_b)

        # Return the supervised dataset
        return state_action_b, target_q_values

    def train(self, pattern_set):
        """
        Update Q-values using pattern set
        """
        # (State, action) and respective target Q-values in a batch
        # random.shuffle(pattern_set)
        # state_action_b, target_q_values = pattern_set
        loss_collection = np.zeros(self.args.agent_epochs)

        # for i in range(self.args.agent_epochs):
        for i in range(10):
            
            random.shuffle(pattern_set)
            state_action_b, target_q_values = zip(*pattern_set[:100])
            # state_action_b, target_q_values = zip(*pattern_set)
            state_action_b = torch.stack(state_action_b)
            target_q_values = torch.stack(target_q_values)

            predicted_q_values = self.net(state_action_b).squeeze()

            loss = nn.functional.mse_loss(predicted_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_collection[i] = loss.item()

        return np.array(loss_collection), loss.item()

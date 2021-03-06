import torch # ml supremacy
import torch.nn as nn
import torch.nn.functional as F # press F to pay respect

import numpy as np

import copy

from agents import RandomAgent

# set the device
device = torch.device("cuda")

# a class that trains a TD3 instance with the given environment
# TODO: - implement actual training step
#       - logging

class Trainer(object):
    def __init__(self, env):
        self.env_template = env
        
        # the td3 instances for training and evaluation
        self.td3 = TD3(self.env_template.state_dim[0], self.env_template.action_dim[0]) # train this instance
        self.prev_td3 = copy.deepcopy(self.td3) # evaluate against the last model version before the train step
        self.random_agent = RandomAgent()

    # evaluates the current policy against the previous one and a random agent
    # returns the amount of won games of n played games
    def evaluate(self, n_games=4):
        # base function for multiprocessing
        rewards = []
        for i in range(n_games):
            # three models -> three players
            env = self.env_template(3, 5)
        
            # our agents
            agents = [
                self.td3,
                self.prev_td3,
                self.random_agent
            ]

            # reset the env and obtain the initial observation
            obs, reward, done = env.reset()
                
            timestep = 0

            # main loop
            while not done:
                timestep += 1
                # get the action mask
                action_mask = env.currentPlayer.getActionMask(env.pullStack, env.playStack)
                    
                # select an action with the current agent
                action = agents[env.currentPlayerID].selectAction(obs, action_mask)
                    
                # perform a step in the environment
                obs, reward, done = env.step(int(action.item()))

                if timestep > 1000:
                    done = True
                
            rewards.append(reward)
            
        return rewards
        
    
    def train(self, n_steps, *env_args):
        # the environment
        env = self.env_template(*env_args)
        # the replaybuffer
        B = ReplayBuffer(self.env_template.state_dim, self.env_template.action_dim, env_args[0])
        
        # reset
        state, reward, done = env.reset()
        episode_reward = []
        episode_timesteps = 0
        episode_num = 0
        
        # fill the whole buffer
        for t in range(int(1e6)):
            
            # retrieve the action mask
            action_mask = env.currentPlayer.getActionMask(env.pullStack, env.playStack)

            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < 25e3:
                action = self.random_agent.selectAction(state, action_mask)

            else:
                action = self.td3.selectAction(state, action_mask)

            # Perform action
            next_state, reward, done,  = env.step(action.item()) 

            done_bool = float(done) if episode_timesteps < 1000 else 0

            # Store data in replay buffer
            B.add(env.currentPlayerID, state, env.cardToTensor(action), next_state, reward, done_bool)
            state = next_state
            episode_reward.append(reward)


            # Train agent after collecting sufficient data
            if t >= 25e3:
                #print("Training...")
                self.td3.train(B, 256)

            # if the episode ends
            if done: 
                # Reset environment
                state, reward, done = env.reset()
                episode_reward = []
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % 5e3 == 0:
                print(self.evaluate())
                
                # copy the trained policy to the prev policy
                self.prev_td3 = copy.deepcopy(self.td3)

#### WARNING: Possible dimension missmatch in the models. Haven't tested these!

# a model containing the policy network ??
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # the underlying model (Conv2d -> Conv2d -> Linear)
        self.model = nn.Sequential(
            nn.Conv2d(state_dim, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(160, 54)
        )

    def forward(self, state):
        return F.softmax(self.model(state))


# a model containing the two Q-Networks Q1 and Q2
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Conv2d(state_dim + action_dim, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(160, 1) # correct dimensions
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Conv2d(state_dim + action_dim, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(160, 1) # correct dimensions
        )

    # returns the output of both Q-Networks
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    # returns the output of Q1
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        return q1

    # returns the output of Q2
    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)
        q2 = self.q2(sa)
        return q2

# WIP -> new concept
# a model responsible to predict the fairness of an action
class Ethicist(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Ethicist, self).__init__()

        # the model architecture
        self.model = nn.Sequential(
            nn.Conv2d(state_dim, action_dim, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(160, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.model(sa)


# Implementation of the Twin Delayed Deep Deterministic Policy Gradient Method (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# GitHub: https://github.com/sfujim/TD3

class TD3(object):
    def __init__(
        self, 
        state_dimension, 
        action_dimension, 
        discount=0.99, 
        tau=0.005, 
        policy_noise=0.2, 
        noise_clip=0.5, 
        policy_frequency=2):
        
        # the actor containing the policy network
        self.actor = Actor(state_dimension, action_dimension).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # the critic containing the two Q-Networks 
        self.critic = Critic(state_dimension, action_dimension).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # hyperparameters
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise 
        self.noise_clip = noise_clip
        self.policy_frequency = policy_frequency
        
        self.total_it = 0
    
    # selects an action given the a state using the current policy
    def selectAction(self, state, actionMask):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        pred = self.actor(state).cpu().data.flatten()
        masked_pred = pred * torch.tensor(actionMask)
        return torch.argmax(masked_pred)
    
    
    # trains the policy and the q networks with the given replay buffer
    def train(self, replayBuffer, batch_size=200):
        self.total_it += 1
        
        # sample from the replay buffer
        state, action, next_state, reward, not_done = replayBuffer.sample(batch_size)
                
        with torch.no_grad():
            # calculate the exploration noise a
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # select the action with the exploration noise a
            next_action = (self.actor_target(next_state).reshape(batch_size, 1, 6, 9) + noise).clamp(0, 1)
            
            # extract the Q-Value of the target Critic using the competing Q-Networks -> prevents overestimating the Q-Value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # calulate the target Q-Value with the observations of the replay buffer
            target_Q = reward + not_done * self.discount * target_Q
            
        
        # optimize current critic
        current_Q1, current_Q2 = self.critic(state, action)
        
        # loss function of the critic
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # optimization step
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # update the policy
        if self.total_it % self.policy_frequency == 0:
            
            # loss function of the actor
            actor_loss = -self.critic.Q1(state, self.actor(state).reshape(batch_size, 1, 6, 9)).mean()
            
            # optimization step
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # copy the optimized parameters from the current networks to the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

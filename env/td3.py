import torch # ml supremacy
import torch.nn as nn
import torch.nn.functional as F # press F to pay respect

import numpy as np

import copy

import multiprocessing as mp

from agents import RandomAgent

# set the device
device = torch.device("cpu")

# a replay buffer class
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, n_players, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.gameID = 0
        self.maxGamesInBuffer = 512
        
        # containers
        self.states = np.zeros((max_size, state_dim[0], state_dim[1], state_dim[2]))
        self.actions = np.zeros((max_size, action_dim[0], action_dim[1], action_dim[2]))
        self.next_states = np.zeros((max_size, state_dim[0], state_dim[1], state_dim[2]))
        self.reward = np.zeros((self.maxGamesInBuffer, n_players, 1))
        self.playerIDs = np.zeros((max_size, 1))
        self.gameIDs = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    # adds a transition tuple to the buffer
    def add(self, playerID, obs, action, next_obs, reward, done):
        self.states[self.ptr] = obs
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_obs
        self.playerIDs[self.ptr] = playerID
        self.not_done[self.ptr] = 1. - done
        self.gameIDs[self.ptr] = self.gameID
        if done:
            for i, r in enumerate(reward):
                self.reward[self.gameID][i] = r
            self.gameID += 1

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # returns a batch of the size batch_size, containing the past transitions
    def sample(self, batch_size):
        # get n random indexes
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(np.array(self.states[ind])).to(device),
            torch.FloatTensor(np.array(self.actions[ind])).to(device),
            torch.FloatTensor(np.array(self.next_states[ind])).to(device),
            torch.FloatTensor(np.array(self.reward[self.gameIDs[ind].astype(int)].flatten()[self.playerIDs[ind].astype(int)].flatten())).to(device),
            torch.FloatTensor(np.array(self.not_done[ind])).to(device)
        )

# a class that trains a TD3 instance with the given environment
# TODO: - implement actual training step
#       - logging

class Trainer(object):
    def __init__(self, env):
        self.env = env
        
        # the td3 instances for training and evaluation
        self.td3 = TD3(self.env.state_dim[0], self.env.action_dim[0]) # train this instance
        self.prev_td3 = copy.deepcopy(self.td3) # evaluate against the last model version before the train step
        self.random_agent = RandomAgent()

    # evaluates the current policy against the previous one and a random agent
    # returns the amount of won games of n played games
    def evaluate(self, n_games=128, num_processes=-1):
        
        # the number of processes to start
        num_processes = mp.cpu_count() if num_processes == -1 else num_processes
        
        # base function for multiprocessing
        def evaluationWorker(n_games):
            rewards = []
            
            for i in range(n_games):
                # three models -> three players
                env = self.env(3, 5)
        
                # our agents
                agents = [
                    self.td3,
                    self.prev_td3,
                    self.random_agent
                ]

                # reset the env and obtain the initial observation
                obs, reward, done = env.reset()
        
                # main loop
                while not done:
                    # get the action mask
                    action_mask = env.currentPlayer.getActionMask(env.pullStack, env.playStack)
                    # select an action with the current agent
                    action = agents[env.currentPlayerID].selectAction(obs, action_mask)
                    
                    # perform a step in the environment
                    obs, reward, done = env.step(action.item())
                
                rewards.append(reward)
            
            return rewards
        
        # container for the processes
        processes = []
        
        # start the processes
        for _ in range(num_processes):
            p = mp.Process(target=evaluationWorker, args=(n_games//num_processes,))
            processes.append(p)
            p.start()

        # join the processes
        for p in processes:
            p.join()
    
    def train(self, n_steps, n_processes, *env_args):
        # the environment
        env = self.env(*env_args)
        # the replaybuffer
        B = ReplayBuffer(self.env.state_dim, self.env.action_dim, self.env.num_players)
        
        # reset
        state, reward, done = env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        
        # fill the whole buffer
        for t in range(int(1e6)):
            
            # retrieve the action mask
            action_mask = env.currenPlayer.getActionMask(env.pullStack, env.playStack, binary=False)

            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < 25e3:
                action = self.random_agent.selectAction(action_mask)

            else:
                action = (
                    policy.selectAction(state)
                    + np.random.normal(0, max_action * 0.1, size=action_dimension)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done,  = env.step(action) 

            #done_bool = float(done) if episode_timesteps < 1000 else 0

            # Store data in replay buffer
            B.add(state, env.cardToTensor(action), next_state, reward, done_bool)
            state = next_state
            episode_reward += reward


            # Train agent after collecting sufficient data
            if t >= 25e3:
                self.td3.train(B, 256)

            # if the episode ends
            if done: 
                # Reset environment
                state, reward, done = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % 5e3 == 0:
                print(self.evaluate())
                
                # copy the trained policy to the prev policy
                self.prev_td3 = copy.deepcopy(self.td3)

#### WARNING: Possible dimension missmatch in the models. Haven't tested these!

# a model containing the policy network π
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
        print(actionMask) 
        masked_pred = pred * torch.tensor(actionMask)
        #print(masked_pred)
        return torch.argmax(masked_pred)
    
    
    # trains the policy and the q networks with the given replay buffer
    def train(self,replayBuffer, batch_size=200):
        self.total_it += 1
        
        # sample from the replay buffer
        state, action, next_state, reward, not_done = replayBuffer.sample(batch_size)
        
        with torch.no_grad():
            # calculate the exploration noise a
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            # select the action with the exploration noise a
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
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
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # optimization step
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # copy the optimized parameters from the current networks to the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

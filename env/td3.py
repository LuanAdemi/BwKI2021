import torch
import torch.nn as nn
import torch.nn.functional as F # press F to pay respect

import numpy as np

device = torch.device("cpu")

# a replay buffer class
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, n_players, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # containers
        self.states = np.zeros((max_size, state_dim[0], state_dim[1], state_dim[2]))
        self.actions = np.zeros((max_size, action_dim[0], action_dim[1], action_dim[2]))
        self.next_states = np.zeros((max_size, state_dim[0], state_dim[1], state_dim[2]))
        self.reward = np.zeros((n_players, 1))
        self.playerIDs = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    # adds a transition tuple to the buffer
    def add(self, playerID, obs, action, next_obs, reward, done):
        self.states[self.ptr] = obs
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_obs
        self.playerIDs[self.ptr] = playerID
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # returns a batch of the size batch_size, containing the past transitions
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(np.array(self.states[ind])).to(device),
            torch.FloatTensor(np.array(self.actions[ind])).to(device),
            torch.FloatTensor(np.array(self.next_states[ind])).to(device),
            torch.FloatTensor(np.array(self.reward[self.playerIDs[ind].astype(int)])).to(device),
            torch.FloatTensor(np.array(self.not_done[ind])).to(device)
		)


# a model containing the policy network π
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(state_dim, 18, 3),
            nn.ReLU(),
            nn.Conv2d(18, 18, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(180, action_dim)
        )
    def forward(self, state):
        return torch.tanh(self.model(state))


# a model containing the two Q-Networks Q1 and Q2
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Conv2d(state_dim + action_dim, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU()
            nn.Flatten(),
            nn.Linear(32, 1) # correct dimensions
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Conv2d(state_dim + action_dim, 32, 3),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32, 1) # correct dimensions
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

    # returns the output of Q1
    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)
        q2 = self.q2(sa)
        return q2


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# GitHub: https://github.com/sfujim/TD3
class TD3(object):
    def __init__(
        self, 
        state_dimension, 
        action_dimension, 
        max_action,
        discount=0.99, 
        tau=0.005, 
        policy_noise=0.2, 
        noise_clip=0.5, 
        policy_frequency=2):
        
        # the actor containing the policy network
        self.actor = Actor(state_dimension, action_dimension, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # the critic containing the two Q-Networks 
        self.critic = Critic(state_dimension, action_dimension).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # hyperparameters
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise 
        self.noise_clip = noise_clip
        self.policy_frequency = policy_frequency
        
        self.total_it = 0
    
    # selects an action given the a state using the current policy
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    
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
        
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # update the policy
        if self.total_it % self.policy_frequency == 0:
            
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # copy the parameters
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Actor-Critic with Experince replay and memory batch a modified version of Actor-Critic algorithm
# Adapted to solve the problem of microgrid energy management
# Mohammad Shabani

import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from environment import *
import random
from matplotlib import pyplot
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MicroGridEnv()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0007

EPS_START = 0.2
EPS_STOP = .001
EPS_DECAY = 3e-3

GAMMA = 0.9
BATCH_SIZE = 5
MEMORY_CAPACITY = 2000

DAY0 = 0
DAYN = 10

TOTAL_REWARDS = []
REWARDS = {}
for i in range(DAY0,DAYN+1):
    REWARDS[i]=[]

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 512)
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.linear3 = nn.Linear(512, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.dropout(output)
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(p=0.05, inplace=False)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.dropout(output)
        value = self.linear3(output)
        return value

class ExperienceReplay:
    def __init__(self):
        self.memory = []
        self.position = 0

    def push(self, experience):
        if len(self.memory) < MEMORY_CAPACITY:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % MEMORY_CAPACITY

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_epsilon(iter):
    return max(EPS_START -  iter * EPS_DECAY, EPS_STOP)

def get_action(actor, state, deterministic=False):
    state = torch.Tensor(state).to(device)
    distribution = actor(state)
    if deterministic:
        # action = distribution.sample().item()
        # action = torch.argmax(distribution.probs).item()
        # action = distribution.sample().item()
        action = np.random.choice(action_size)
    else:
        # action = np.random.choice(action_size)
        action = distribution.sample().item()
    return action

def compute_returns(next_value, rewards, dones, gamma=0.99):
    R = next_value
    for step in reversed(range(len(rewards))):
        R = rewards + gamma * R * (1 - dones[step])
    return R

def update(actor, critic, actor_optimizer, critic_optimizer, experiences):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    for experience in experiences:
        state, action, reward, next_state, done = experience
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
    
    # Compute next state values
    values = critic(states)
    next_values = critic(next_states)
    
    returns = compute_returns(next_values, rewards, dones)
    
    advantage = returns - values

    # Compute critic loss
    critic_loss = advantage.pow(2).mean()

    # Compute actor loss
    
    distribution = actor(states)
    log_probs = distribution.log_prob(actions)
    actor_loss = -(log_probs * advantage.detach()).mean()

    # Update networks
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    actor_optimizer.step()
    critic_optimizer.step()

def trainIters(actor, critic, n_iters, day=None):
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    experience_replay = ExperienceReplay()

    for iter in range(n_iters):
        state = env.reset(day0=DAY0,dayn=DAYN, day=None)
        total_reward = 0
        done = False

        while not done:
            eps = get_epsilon(iter)
            # print(eps)
            action = get_action(actor, state, deterministic=random.random() < eps)

            next_state, reward, done = env.step(action)
            total_reward += reward

            experience_replay.push((state, action, reward, next_state, done))
            state = next_state

            if len(experience_replay) >= BATCH_SIZE:
                experiences = experience_replay.sample(BATCH_SIZE)
                update(actor, critic, actor_optimizer, critic_optimizer, experiences)

        print('Iteration: {}, Reward: {}'.format(iter, total_reward))
        REWARDS[env.day].append(total_reward)

    env.close()

if __name__ == '__main__':
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)

    if os.path.exists('model/critic.pkl'):
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)

    trainIters(actor, critic, n_iters=100)
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    print("Training finished")
    # Creating the plot
    total_reward = []
    
    for rew in REWARDS.values():
        # print(np.average(list(rew)))
        pyplot.plot(list(rew))
    pyplot.legend(["Day {}".format(i) for i in range(11)], loc = 'upper right')
    pyplot.show()
    
    sum_rewards = 0
    min_len = 2000
    
    # print("average= ",np.average([list(REWARDS[i])[-1] for i in range(DAY0,DAYN)]))


    for rew in REWARDS.values():
        if len(rew) < min_len:
            min_len = len(rew)

    print(min_len)

    for i in range(min_len):
        for rew in REWARDS.values():
            sum_rewards += rew[i]
        TOTAL_REWARDS.append(sum_rewards / 6)
        sum_rewards = 0
    
    SAVE_REWARD = []
    
    for rew in REWARDS.values():
        SAVE_REWARD.append(np.mean(rew[min_len - 12:min_len - 1]))
        
    import pickle
    with open("REWARDS_ACTORCRITIC++.pkl", 'wb') as f:
        pickle.dump(SAVE_REWARD,f,pickle.HIGHEST_PROTOCOL)
        

    pyplot.plot(list(TOTAL_REWARDS))
    pyplot.show()
    
    
    state = env.reset(day0=DAY0,dayn=DAYN, day=None)
    done = False
    while True:
        # Pick an action from the action space (here we pick an index between 0 and 80)
        # action = env.action_space.sample()
        # action =[np.argmax(action[0:4]),np.argmax(action[4:9]),np.argmax(action[9:11]),np.argmax(action[11:])]
        action = get_action(actor, state)
        # Using the index we get the actual action that we will send to the environment
        # print(ACTIONS[action])
        print(action)
        # Perform a step in the environment given the chosen action
        # state, reward, terminal, _ = env.step(action)
        state, reward, terminal = env.step(list(action))
        env.render()
        print(reward)
        rewards.append(reward)
        if terminal:
            break
    
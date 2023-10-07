# Actor-Critic 
# Adapted to solve the problem of microgrid energy management
# Mohammad Shabani

import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from environment import *
import random as random
from matplotlib import pyplot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MicroGridEnv()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001

EPS_START = 0.1
EPS_STOP = .004
EPS_DECAY = 5e-3

DAY0 = 0
DAYN = 1

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
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
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
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters, eps_start, eps_decay, eps_end):
    
    def getEpsilon(iter):
        return max(eps_start -  iter * eps_decay, eps_end)  # linearly interpolate
    
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        
        # for day in range(DAYN - DAY0):
        state = env.reset(day0=DAY0,dayn=DAYN, day=None)
        entropy = 0
        log_probs = []
        values = []
        rewards = torch.zeros(1)
        masks = []
        entropy = 0

        while True:
            # env.render()
            state = torch.Tensor(state).to(device)
            dist, value = actor(state), critic(state)
 
            eps = getEpsilon(iter = iter)
            # print(value)
            
            if random.random() < eps:
                # print('random action')
                action_probs = torch.rand(action_size)  # Random action probabilities using uniform distribution
                action = torch.argmax(action_probs)
            else:
                # action = dist.sample()
                action_probs = torch.rand(action_size)  # Random action probabilities using uniform distribution
                action = torch.argmax(action_probs)
            
            next_state, reward, done = env.step(action.cpu().numpy())
            
            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards  += reward
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            

            state = next_state

            if done:
                
                print('Iteration: {}, Reward: {}'.format(iter, rewards[0]))
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

        REWARDS[env.day].append(rewards.numpy().item())

    # torch.save(actor, 'model/actor.pkl')
    # torch.save(critic, 'model/critic.pkl')
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
    trainIters(actor, critic, n_iters=150, eps_start=EPS_START, eps_decay=EPS_DECAY, eps_end=EPS_STOP)
    print("Training finished")
    
    for rew in REWARDS.values():
        # print(np.average(list(rew)))
        pyplot.plot(list(rew))
    pyplot.legend(["Day {}".format(i) for i in range(11)], loc = 'upper right')
    pyplot.show()
    
    sum_rewards = 0
    min_len = 10000

    for rew in REWARDS.values():
        if len(rew) < min_len:
            min_len = len(rew)

    print(min_len)
    import pickle
    with open("REWARDS_ACTORCRITIC.pkl", 'wb') as f:
        pickle.dump(REWARDS,f,pickle.HIGHEST_PROTOCOL)

    for i in range(min_len):
        for rew in REWARDS.values():
            sum_rewards += rew[i]
        TOTAL_REWARDS.append(sum_rewards / 6)
        sum_rewards = 0

    pyplot.plot(list(TOTAL_REWARDS))
    pyplot.show()
    
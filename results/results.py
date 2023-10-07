from matplotlib import pyplot
import numpy as np
import pickle


sum_rewards = 0
min_len = 2000
TOTAL_REWARDS_SARSA = []
TOTAL_REWARDS_DQN = []
TOTAL_REWARDS_PPO = []
TOTAL_REWARDS_AC = []

count = 0
for i in range(10):
    try:
        rewards=pickle.load(open('REWARDS_SARSA' + str(i) + '.pkl','rb'))
    except:
        break
    else:
        count += 1
        for rew in rewards.values():
            if len(rew) < min_len:
                min_len = len(rew)
for i in range(min_len):
    for j in range(count):
        rewards=pickle.load(open('REWARDS_SARSA' + str(j) + '.pkl','rb'))
        for rew in rewards.values():
            sum_rewards += rew[i]
    TOTAL_REWARDS_SARSA.append(sum_rewards / (count * 10))
    sum_rewards = 0
pyplot.plot(list(TOTAL_REWARDS_SARSA), label='SARSA')
pyplot.legend()
count = 0  

for i in range(10):
    try:
        rewards=pickle.load(open('REWARDS_actorcritic++' + str(i) + '.pkl','rb'))
    except:
        break
    else:
        count += 1
        for rew in rewards.values():
            if len(rew) < min_len:
                min_len = len(rew)
for i in range(min_len):
    for j in range(count):
        rewards=pickle.load(open('REWARDS_actorcritic++' + str(j) + '.pkl','rb'))
        for rew in rewards.values():
            sum_rewards += rew[i]
    TOTAL_REWARDS_AC.append(sum_rewards / (count * 10))
    sum_rewards = 0
pyplot.plot(list(TOTAL_REWARDS_AC), label='actor critic')
pyplot.legend()
count = 0  

for i in range(10):
    try:
        rewards=pickle.load(open('REWARDS_DQN' + str(i) + '.pkl','rb'))
    except:
        break
    else:
        count += 1
        for rew in rewards.values():
            if len(rew) < min_len:
                min_len = len(rew)
for i in range(min_len):
    for j in range(count):
        rewards=pickle.load(open('REWARDS_DQN' + str(j) + '.pkl','rb'))
        for rew in rewards.values():
            sum_rewards += rew[i]
    TOTAL_REWARDS_DQN.append(sum_rewards / (count * 10))
    sum_rewards = 0
pyplot.plot(list(TOTAL_REWARDS_DQN), label='DQN')
pyplot.legend()
count = 0  

for i in range(10):
    try:
        rewards=pickle.load(open('REWARDS_PPO' + str(i) + '.pkl','rb'))
    except:
        break
    else:
        count += 1
        for rew in rewards.values():
            if len(rew) < min_len:
                min_len = len(rew)
for i in range(min_len):
    for j in range(count):
        rewards=pickle.load(open('REWARDS_PPO' + str(j) + '.pkl','rb'))
        for rew in rewards.values():
            sum_rewards += rew[i]
    TOTAL_REWARDS_PPO.append(sum_rewards / (count * 10))
    sum_rewards = 0
pyplot.plot(list(TOTAL_REWARDS_PPO), label='PPO')
pyplot.legend()
count = 0  

pyplot.show()

TOTAL_REWARDS_AC = []

for i in range(10):
    try:
        rewards=pickle.load(open('REWARDS_ACTORCRITIC' + str(i) + '.pkl','rb'))
    except:
        break
    else:
        count += 1
        for rew in rewards.values():
            if len(rew) < min_len:
                min_len = len(rew)
for i in range(min_len):
    for j in range(count):
        rewards=pickle.load(open('REWARDS_ACTORCRITIC' + str(j) + '.pkl','rb'))
        for rew in rewards.values():
            sum_rewards += rew[i]
    TOTAL_REWARDS_AC.append(sum_rewards / (count * 10))
    sum_rewards = 0
pyplot.plot(list(TOTAL_REWARDS_AC), label='ACTORCRITIC')
pyplot.legend()
count = 0  

pyplot.show()
    

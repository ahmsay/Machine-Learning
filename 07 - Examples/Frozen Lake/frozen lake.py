from IPython.display import HTML
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/q2ZOEFAaaI0?showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')

import numpy as np
import gym
import random

env = gym.make("FrozenLake-v0")

action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))
print(qtable)

total_episodes = 15000  # total episodes
learning_rate = 0.8     # learning rate
max_steps = 99          # max steps per episode
gamma = 0.95            # discounting rate
# exploration parameters
epsilon = 1.0           # exploration rate
max_epsilon = 1.0       # exploration probability at start
min_epsilon = 0.01      # minimum exploration probability
decay_rate = 0.005      # exponential decay rate for exploration prob

rewards = []

for episode in range(total_episodes):
    # reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        # choose an action a in the current world state (s)
        # first we randomize a number
        exp_exp_tradeoff = random.uniform(0,1)
        # if this number > epsilon --> exploitation (taking the biggest q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        # else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
            
        # take the action (a) and observe the outcome state (s') and reward (r)
        new_state, reward, done, info = env.step(action)
        
        # update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s, a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_rewards += reward
        
        # our new state is state
        state = new_state
        
        # if done (if we're dead): finish episode
        if done == True:
            break
        
    # reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards)/total_episodes))
print(qtable)


env.reset()

for episode in range(20):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
            
            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
env.close()
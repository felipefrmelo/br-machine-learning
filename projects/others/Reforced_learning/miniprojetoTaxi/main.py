from agent import Agent
from monitor import interact
import gym
import numpy as np

from scipy.optimize import minimize

env = gym.make('Taxi-v2')
def maxx(x):

    
    agent = Agent( alpha=x[0], eps =  x[1] ,gamma= x[2])
    avg_rewards, best_avg_reward = interact(env, agent, num_episodes=500)
    print('best_avg={:e} || alpha={:e} || eps={:e} || gamma={:e}'.format(best_avg_reward,x[0],x[1],x[2]))
    return -best_avg_reward


bnds = tuple( [(0,1) for i in range(3)] )

x0 = [0.9,1e-5,1.0]

#res = minimize(maxx, x0, bounds=bnds, options={'disp':True})
import gym
env = gym.make('SpaceInvaders-v0')
env.reset()
env.render()
from agent import Agent
from qdl_agent import QDl_Agent
from monitor import interact
import gym
import numpy as np
import os
import time
env = gym.make('Taxi-v2')

env.seed(0)
agent = Agent(nA = 6 ,alpha=0.2, eps =  1e-4 ,gamma= 1)

#avg_rewards, best_avg_reward = interact(env, agent, num_episodes=500, verbose= True)

agent_q = QDl_Agent(state_size = 500 ,action_size = 6, seed =0, alpha=0.2, gamma= .91)
print()
avg_rewards, best_avg_reward = interact(env, agent_q, num_episodes=500, verbose= True)

# observation = env.reset()
# best_score = -np.inf
# for i in range(300):
    
    
#     score = 0
#     for j in range(1,32):
#         print( "Tentativa ",i+1)
#         env.render()
#         action = agent.select_action(observation) # your agent here (this takes random actions)
#         observation, reward, done, info = env.step(action)
#         score += reward 
#         print('Step: {}\nBest score: {}'.format(j,best_score))
#         time.sleep(0.5)
#         if score > best_score:
#             best_score = score
        
#         os.system('clear')
#         if done:
            
#             observation = env.reset()
#             break

#     _, _ = interact(env_treino, agent, num_episodes=j+6048)
#     env.reset()
#     env.close()
    
   
    


# def maxx(x):

    
#     agent = Agent( alpha=x[0], eps =  x[1] ,gamma= x[2])
#     avg_rewards, best_avg_reward = interact(env, agent, num_episodes=500)
#     print('best_avg={:e} || alpha={:e} || eps={:e} || gamma={:e}'.format(best_avg_reward,x[0],x[1],x[2]))
#     return -best_avg_reward


# bnds = tuple( [(0,1) for i in range(3)] )

# x0 = [0.9,1e-5,1.0]

#res = minimize(maxx, x0, bounds=bnds, options={'disp':True})

import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha = 1e-2, gamma = 1, eps = 0.02 ):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = eps
        self.episode = 0

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        qs = self.Q[state]  

        p = self.get_prop(qs)

        action = np.random.choice(np.arange(self.nA), p = p)

        return action
    
    def get_prop(self, qs):
        
        self.epsilon = self.epsilon(self.episode) if callable(self.epsilon) else self.epsilon
        p = np.ones(self.nA) * self.epsilon /self.nA
        best_a = np.argmax(qs)
        p[best_a] = 1 - self.epsilon + ( self.epsilon / self.nA)
        return p

    def update_Q(self, Qsa, Qsa_next, reward):
        """ updates the action-value function estimate using the most recent time step """
        
        return Qsa + (self.alpha * (reward + (self.gamma * Qsa_next) - Qsa))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        
        p = self.get_prop(self.Q[next_state])

        self.Q[state][action] = self.update_Q(self.Q[state][action],np.dot(p, self.Q[next_state]) , reward) 
        
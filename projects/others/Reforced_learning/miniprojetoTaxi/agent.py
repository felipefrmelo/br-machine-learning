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
        self.Q1 = defaultdict(lambda: np.zeros(self.nA))
        self.Q2 = defaultdict(lambda: np.zeros(self.nA))
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

        qs = self.Q1[state] + self.Q2[state] 

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
        
        
        if np.random.randint(2) == 0:
            self.Q1[state][action] = self.update_Q(self.Q1[state][action], self.Q2[next_state][np.argmax(self.Q1[next_state])], reward) 
        else:
            self.Q2[state][action] = self.update_Q(self.Q2[state][action], self.Q1[next_state][np.argmax(self.Q2[next_state])], reward)

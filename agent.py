from abc import ABC, abstractmethod
import os
import pickle
import collections
import numpy as np
import random
from copy import deepcopy



class Learner(ABC):
    def __init__(self, alpha, gamma, eps, eps_decay=0., encourage_explore=True):
        # Agent parameters
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.prev_action = None
        self.prev_state = None
        self.encourage_explore = encourage_explore
        
        self.actions = []
        # 9 possible actions for 9 grids
        for row in range(3):
            for col in range(3):
                self.actions.append((row,col))
        
        # Q value table, the table is 9x512, 9 actions and 512 unqiue states
        self.Q_table={}
        for action in self.actions:
            self.Q_table[action] = collections.defaultdict(int) # every action contain a dict

        self.rewards = []
        
    def get_action(self, state):
        ''' 
        Make an action for the current state with the largest Q-value
        State: a string contain the state
        '''
        self.prev_state = deepcopy(state)
        # check the 1D string state and return 2D position
        possible_actions = [a for a in self.actions if state[a[0]*3 + a[1]] == ' ']
        if random.random() < self.eps: # add some randomness, so that your competitor cannot guess your next action
            action = random.choice(possible_actions)
        else:
            # values is a 9 dim vector because each unique state has 9 actions
            # the initial Q(state, action) value is 0
            q_list = np.array([self.getQ(action, state) for action in possible_actions])
            maxQ = np.where(q_list == np.max(q_list))[0]

            if len(maxQ)>1: # more than 1 max q value
                ix_select = np.random.choice(maxQ, 1)[0]
            else:
                ix_select = maxQ[0]
            action = possible_actions[ix_select] # pick the action with largest Q value

        self.eps *= (1-self.eps_decay) # update epsilon
        self.prev_action = deepcopy(action)
        return action
        
    def save(self, path):
        ''' Save the agent '''
        if os.path.isfile(path):
            os.remove(path)
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()
        
    @abstractmethod
    def update(self, next_state, next_action, reward):
        pass
    @abstractmethod
    def getQ(self):
        pass


class QLearner(Learner):
    def __init__(self, alpha, gamma, eps, eps_decay=0., encourage_explore=True):
        super().__init__(alpha, gamma, eps, eps_decay, encourage_explore)
    
    def update(self, next_state, next_action, reward):
        ''' Update the Q table with the action-value function Q (Bellman Operator) '''
        # check all possible actions in next state
        if self.prev_state is not None:
            previous_Q = self.getQ(self.prev_action, self.prev_state)
            possible_actions = [a for a in self.actions if self.prev_state[a[0]*3 + a[1]] == ' ']
            values = np.array([self.getQ(action, next_state) for action in possible_actions])
            '''print(possible_actions)
            print(self.prev_state)
            print(next_state)
            #print(max(values))
            print('reward:', reward)'''
            self.Q_table[self.prev_action][self.prev_state] = previous_Q + self.alpha*((reward + self.gamma*max(values))-previous_Q)
            self.rewards.append(reward)
    
    
    def getQ(self, action, state):
        if self.Q_table[action][state] == 0 and self.encourage_explore:
            #if random.random() < 0.5: # have 50% chance to explore new action-state
            self.Q_table[action][state] = 1.0 # encourage the agent to explore other action-state
        return self.Q_table[action][state]
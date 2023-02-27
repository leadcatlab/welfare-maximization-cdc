import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import seaborn as sns

class Component:

    def __init__(self, name='test', initial_state=100, inspect_cost=None, replace_cost=None, importance_score=None, dynamics_scale=10, dynamics_shape=1.5):

        # component parameters
        self.name = name
        self.num_states = 101
        self.num_obs = 102
        self.num_actions = 3
        self.states = np.arange(0, self.num_states, 1)
        self.obs = np.arange(0, self.num_obs, 1)
        self.actions = np.arange(0, self.num_actions, 1)
        self.failure_condition = 0
        self.trans_prob = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.obs_prob = np.zeros((self.num_states, self.num_actions, self.num_obs))        
        self.initial_state = initial_state
        if importance_score is None:
            self.i_score = 1.0
        else:
            self.i_score = importance_score

        # component logging
        self.current_state = initial_state
        self.num_steps = 0
        self.state_history = []
        self.obs_history = []
        self.action_history = []
        self.offline_count = 0
        self.offline_steps = []
        self.cost_history = []
        if inspect_cost is None:
            self.inspect_cost = 10
        else:
            self.inspect_cost = inspect_cost
        if replace_cost is None:
            self.replace_cost = 50
        else:
            self.replace_cost = replace_cost

        # synthesize data
        self.dynamics_scale = dynamics_scale
        self.dynamics_shape = dynamics_shape
        self.trans_prob = self.gen_trans_prob()
        self.obs_prob = self.get_obs_prob()
        
    def reset(self):
        """
        Reset the component state and histories
        """
        self.current_state = self.initial_state
        self.num_steps = 0
        self.state_history = [self.current_state]
        self.obs_history = []
        self.action_history = []
        self.cost_history = []
        self.offline_count = 0
        self.offline_steps = []

    def synthesize_dynamics(self, state):
        """
        Synthesize transition dynamics from the given state in the absence of maintenance action
        """
        probs = np.zeros(self.num_states)
        for next_state in range(state,-1,-1):
            probs[next_state] = weibull_min.pdf(101-next_state+1, self.dynamics_shape , scale=self.dynamics_scale, loc=101-state)
        probs = probs/np.sum(probs)
        return probs

    def gen_trans_prob(self):
        """
        A function that generates transition probability for states of the component
        """

        trans_prob = np.zeros((self.num_states, self.num_actions, self.num_states))
        for state in self.states:
            for action in self.actions:

                # no reload action or inspection action
                if action in [0,1]:
                    trans_prob[state, action, :] = self.synthesize_dynamics(state)

                # replace action
                elif action == 2:
                    trans_prob[state, action, self.states[-1]] = 1.0
                    
        return trans_prob

    def get_obs_prob(self):
        """
        A function that generates observation probability of the health of the component based on 
        a given distribution

        Accounts for chances of inspector human bias and variability in health measurement
        """

        obs_prob = np.zeros((self.num_states, self.num_actions, self.num_obs))
        for state in self.states:
            for action in self.actions:

                # other observation 101 for no action or replace action
                if action in [0,2]:
                    obs_prob[state, action, 101] = 1.0
                
                # inspection action
                #TODO: make this more realistic
                elif action == 1:  
                    if state > 2 and state < 98:
                        obs_prob[state, action, state] = 0.7
                        obs_prob[state, action, state+1] = 0.1
                        obs_prob[state, action, state-1] = 0.1
                        obs_prob[state, action, state+2] = 0.05
                        obs_prob[state, action, state-2] = 0.05
                    else:
                        obs_prob[state, action, state] = 1.0
        return obs_prob
       
    def get_failure_probability(self, given_state=None):
        """
        Calculate the failure probability using the given state, failure threshold and the transition probability. 
        If no state is given, use current state
        """
        
        if given_state is None:
            given_state = self.current_state
        given_state = int(given_state)
        failure_prob = 0.0
        for state in self.states:
            if state <= self.failure_condition:
                failure_prob += self.trans_prob[given_state, 0, state]
        return failure_prob

    def update(self, action):
        """
        Update the state of the component and log data
        """
        
        self.current_state = np.random.choice(self.num_states, p=self.trans_prob[self.current_state, action, :])
        obs = np.random.choice(self.num_obs, p=self.obs_prob[self.current_state, action, :])

        if self.current_state <= self.failure_condition:
            self.current_state = self.failure_condition
            self.offline_count += 1
            self.offline_steps.append(self.num_steps)

        if action == 0:
            self.cost_history.append(0)
        elif action == 1:
            self.cost_history.append(self.inspect_cost)
        elif action == 2:
            self.cost_history.append(self.replace_cost)
        
        self.num_steps += 1
        self.obs_history.append(obs)
        self.state_history.append(self.current_state)
        self.action_history.append(action)

    def visualize_history(self):
        """
        Visualize the historic state of the component and the action taken
        """
        get_inspect_indices = [i for i in range(1,len(self.action_history)) if self.action_history[i] == 1]
        get_replace_indices = [i for i in range(1,len(self.action_history)) if self.action_history[i] == 2]
        get_inspect_states = [self.state_history[i] for i in get_inspect_indices]
        get_replace_states = [self.state_history[i] for i in get_replace_indices]

        plt.figure(figsize=(15,5))
        plt.plot(self.state_history, '.-', color='gray', linewidth=2, alpha=1.0,  label='component CI')
        plt.scatter(get_inspect_indices, get_inspect_states, marker='o', color='green', s=70, alpha=0.8, label='inspection')
        plt.scatter(get_replace_indices, get_replace_states, marker='o', color='red', s=70, alpha=0.8, label='replacement')
        plt.plot(self.failure_condition*np.ones(len(self.state_history)), 'k--', label='failure condition', linewidth=2.5, alpha=0.6)
        plt.xlim(0, len(self.state_history))
        plt.ylim(0, max(self.state_history)+10)
        plt.legend()
        plt.xlabel('time step', fontsize=18)
        plt.xticks(fontsize=14)
        plt.ylabel('condition index', fontsize=18)
        plt.yticks(fontsize=14)
        plt.legend(prop={'size': 14}, frameon=False)
        sns.despine(right=True, top=True)
        plt.title(f'CI History for {self.name} with $\lambda$: {self.i_score}, Inspect Cost: {self.inspect_cost}, Replace Cost: {self.replace_cost}', fontsize=16)
        plt.tight_layout()
        plt.show()

    def visualize_trajectory_samples(self, num_samples=10):
        """
        Visualize a few samples of the trajectory of the component
        """
        trajectory_samples = []
        for i in range(num_samples):
            state_history = []
            action_history = []
            current_state = self.current_state
            while current_state > 0:
                action = 0
                state_history.append(current_state)
                action_history.append(action)
                current_state = np.random.choice(self.states, p=self.trans_prob[current_state, action, :])
            trajectory_samples.append(state_history)

        plt.figure(figsize=(15,5))
        for i in range(num_samples):
            plt.plot(trajectory_samples[i], linewidth=2, alpha=0.8)
        plt.plot(self.failure_condition*np.ones(len(self.state_history)), 'k--', label='failure condition', linewidth=2.5)
        plt.xlabel('time step', fontsize=18)
        plt.xticks(fontsize=14)
        plt.ylabel('Condition Index', fontsize=18)
        plt.yticks(fontsize=14)
        
    def visualize_transition_prob(self, state):
        """
        Visualize the transition probability for a given state for different actions
        """

        actions = [0,1,2]
        colors = ['blue', 'green', 'red']
        
        # figure with three subplots for three actions
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        fig.suptitle(f'Transition Probability for {self.name} at state {state}', fontsize=16)
        for i, action in enumerate(actions):
            dist = self.trans_prob[state, action, :]
            ax[i].bar(self.states, dist, width=0.8, color=colors[i], alpha=0.8)
            ax[i].set_title(f'Action: {action}', fontsize=14)
            ax[i].set_xlabel('State')
            ax[i].set_ylabel('Probability')


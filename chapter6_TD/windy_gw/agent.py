import numpy as np
from collections import defaultdict

class Agent:
  def __init__(self, alpha, epsilon, gamma, diagonal=False, ninth_action=False):
    self.alpha = alpha # Step size
    self.epsilon = epsilon # e-greedy policy
    self.gamma = gamma # discount factor

    # Define action space
    if not diagonal and not ninth_action:
      self.action_space = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    elif diagonal and not ninth_action:
      self.action_space = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    elif diagonal and ninth_action:
      self.action_space = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1), (0, 0)]

    self.valid_actions = defaultdict(list)

  def initialize_qvalues(self):
    # Set default q values to 0
    self.Q = defaultdict(int)
  
  def select_action(self, gw_row_limit, gw_col_limit, state, policy='e-greedy'):
    '''
    Generate valid actions for each state. Select actions based on a policy.
    '''

    q_s_a = []
    # For a given state, generate list of valid actions
    # if it already doesn't exist and get q-values.
    if state not in self.valid_actions:
      for a in self.action_space:
        possible_next_state = np.add(state, a)
        if (0 <= possible_next_state[0] < gw_row_limit) and (0 <= possible_next_state[1] < gw_col_limit):
          q_s_a.append(self.Q[state, a])
          self.valid_actions[state].append(a)
    else:
      # For a given state, get q-values for already known valid actions.
      for a in self.valid_actions[state]:
        q_s_a.append(self.Q[state, a])

    # Selection actions based on e-greedy policy
    if policy == 'e-greedy':
      if np.random.binomial(1, self.epsilon) == 1:
        return self.valid_actions[state][np.random.choice(len(self.valid_actions[state]))]
      else:
        return self.valid_actions[state][np.argmax(q_s_a)]

    # Select actions based on deterministic greedy policy
    elif policy == 'greedy':
      return self.valid_actions[state][np.argmax(q_s_a)]

  def update(self, state, action, reward, next_state, next_action, next_state_terminal=False):
    '''
    Update q-values for a given state-action pair
    '''
    if next_state_terminal:
      self.Q[state, action] += self.alpha * (reward - self.Q[state, action])
    else:
      self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])
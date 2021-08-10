import numpy as np
from tqdm import tqdm

class Sarsa:
  def __init__(self, agent, env, num_episodes):
    self.agent = agent
    self.env = env
    self.num_episodes = num_episodes

  def run_episode(self, policy='e-greedy'):
    '''
    Traverse and learn through an episode based on a specific policy
    '''
    t = 0
    state = (3, 0) # Initial state
    terminal_state_reached = False

    action = self.agent.select_action(self.env.num_rows, self.env.num_cols, state, policy=policy)
    
    while not terminal_state_reached:
      next_state, reward = self.env.step(state, action)
      next_action = self.agent.select_action(self.env.num_rows, self.env.num_cols, next_state, policy=policy)

      if next_state == self.env.goal_pos:
        terminal_state_reached = True
      
      self.agent.update(state, action, reward, next_state, next_action, next_state_terminal=terminal_state_reached)
      state = next_state
      action = next_action

      if policy == 'greedy':
        # Calculating time steps it takes for a deterministic greedy policy
        # to reach terminal state
        t += 1
    
    return t

  def run(self):
    time_steps = []

    self.agent.initialize_qvalues()
    for ep in range(self.num_episodes):

      self.run_episode() # Train using e-greedy
      t = self.run_episode(policy='greedy') # Evaluate greedy policy
      time_steps.append(t)
    
    return time_steps
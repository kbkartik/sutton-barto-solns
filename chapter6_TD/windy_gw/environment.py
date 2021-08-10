import numpy as np

class Environment:
  def __init__(self, num_rows, num_cols, wind_pos, stochastic_wind=False, seed=10):
    self.set_seed(seed)

    # Gridworld parameters
    self.num_rows = num_rows
    self.num_cols = num_cols
    self.start_pos = (int(num_rows/2), 0)
    self.goal_pos = (int(num_rows/2), num_cols-3)
    self.wind_pos = wind_pos
    self.stochastic_wind = stochastic_wind

    self.reward = -1 # Reward distribution

  def set_seed(self, seed):
    np.random.seed(seed)

  def step(self, state, action):
    '''
    Step into the environment and get the next state and reward
    '''
    new_state = np.add(state, action)
    updated_new_state = (new_state[0] - self.wind_pos[new_state[1]], new_state[1]) # Wind factor taken into account

    if self.stochastic_wind and self.wind_pos[new_state[1]] > 0:
      # Random wind factor taken into account
      random_factor = np.random.randint(-1, 2)
      updated_new_state = (updated_new_state[0] - random_factor, updated_new_state[1])

    # Ensuring new state is within environment bounds
    shifted_state = (max(0, min(updated_new_state[0], self.num_rows-1)), max(0, min(updated_new_state[1], self.num_cols-1)))

    assert (0 <= shifted_state[0] < self.num_rows) and (0 <= shifted_state[1] < self.num_cols), print('new state not within env bounds')

    if shifted_state == self.goal_pos:
      # Terminal state reached
      return shifted_state, 0

    return shifted_state, self.reward
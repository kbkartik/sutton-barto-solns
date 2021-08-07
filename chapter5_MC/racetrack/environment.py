import numpy as np
from racetrack import Racetrack

class Environment(Racetrack):

  def __init__(self, rows, cols, min_velocity, max_velocity, racetrack_filepath):
    super().__init__(rows, cols, racetrack_filepath=racetrack_filepath)

    # Min and maximum vehicle velocity
    self.min_velocity = min_velocity
    self.max_velocity = max_velocity
    self.rest_velocity = (0, 0)
    
    # List of all valid vehicle velocities
    self.vehicle_velocity = [(vx, vy) for vx in range(self.min_velocity, self.max_velocity+1) for vy in range(self.min_velocity, self.max_velocity+1)]
    self.initialize_states()

    self.reward = [-1, -20]

  def initialize_states(self):
    '''
    Setting up a dictionary of possible states.

    In the racetrack:
      1) All 0's represent 'out-of-racetrack' positions
      2) All 1's represent initial vehicle positions
      3) All 2's represent intermediate vehicle positions
      4) All 3's represent final vehicle positions
    '''
    
    self.racetrack_positions = [tuple(pos) for pos in np.argwhere(self.racetrack != 0)]
    self.init_racetrack_positions = [tuple(pos) for pos in np.argwhere(self.racetrack == 1)]
    self.final_racetrack_positions = [tuple(pos) for pos in np.argwhere(self.racetrack == 3)]

    # All possible states a vehicle can visit. Excluding intermediate states with velocity 0.
    self.states = []
    for pos in self.racetrack_positions:
      for v in self.vehicle_velocity:
        if (v == self.rest_velocity and self.racetrack[pos] == 2):
          continue
        else:
          self.states.append(tuple(pos) + v)
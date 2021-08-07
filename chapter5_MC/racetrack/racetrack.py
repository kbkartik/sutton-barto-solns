import numpy as np

class Racetrack:
  '''
  Generate a random racetrack of any shape. Any racetrack generated using this code 
  will follow a specific pattern: initial vehicle positions will be on the last row
  and final vehicle positions will be on the last column.

  In the racetrack:
      1) All 0's represent 'out-of-racetrack' positions
      2) All 1's represent initial vehicle positions
      3) All 2's represent intermediate vehicle positions
      4) All 3's represent final vehicle positions
  ''' 

  def __init__(self, rows, cols, racetrack_filepath=None):

    # Dimensions of the racetrack
    self.rows = rows
    self.cols = cols
    self.racetrack = np.zeros((self.rows, self.cols), dtype=int)

    if racetrack_filepath is not None:
      self.upload_racetrack(racetrack_filepath)
    else:
      self.create_racetrack()

  def upload_racetrack(self, racetrack_filepath):
    '''
    Upload a racetrack (numpy file) if available.
    '''
    self.racetrack = np.load(racetrack_filepath)

  def generate_track_boundaries(self, start_pos_x, start_pos_y, final_pos_x, final_pos_y, prob):
    
    # These specific movement choices are used 
    # to create racetrack boundaries.
    movement_choices = [[-1, 0], [0, 1]]

    start_pos = (start_pos_x, start_pos_y)
    final_pos = (final_pos_x, final_pos_y)

    x = start_pos_x
    y = start_pos_y

    while (x != final_pos_x) or (y != final_pos_y):
      incorrect_move = True
      while incorrect_move:
        if (x == start_pos_x and y == start_pos_y) or (x > final_pos_x and y == self.cols - 5):
          move = [-1, 0]
        elif (x == final_pos_x and y > 0) or (x > final_pos_x and y == 0):
          move = [0, 1]
        else:
          move = movement_choices[np.random.binomial(1, prob)]

        pos = (x + move[0], y + move[1])
        x, y = pos

        # Fill intermediate racetrack positions as '2'
        if 0 <= x < self.rows and 0 <= y < self.cols:
          incorrect_move = False
          if self.racetrack[pos] != 2 and self.racetrack[pos] == 0:
            self.racetrack[pos] = 2

  def finalize_racetrack(self):
    '''
    The boundaries of the racetrack have been created. Now, we need
    to fill the gaps inside the racetrack with '2', which denotes intermediate
    racetrack positions.
    '''

    for i in range(self.rows):
      twos_idx = np.argwhere(self.racetrack[i, :] == 2)
      threes_idx = np.argwhere(self.racetrack[i, :] == 3)
      if len(twos_idx) > 0 or len(threes_idx) > 0:
        min_idx = np.min(twos_idx)
        max_idx = np.max(np.concatenate((twos_idx, threes_idx)))
        for j in range(min_idx+1, max_idx):
          pos = (i, j)
          if self.racetrack[pos] == 0:
            self.racetrack[pos] = 2

  def create_racetrack(self):

    init_row = self.rows - 1
    final_col = self.cols - 1

    initial_start_col = 3 # Column number where the first initial position starts
    final_start_row = 0 # Row number where the first final position starts

    # 0.155 is the probability of create a positions up than right for the left boundary
    self.generate_track_boundaries(init_row, initial_start_col, final_start_row, final_col, 0.155)

    initial_stop_col = 8 # Column number where the last initial position stops
    final_stop_row = 5 # Row number where the last final position ends

    # 0.155 is the probability of create a positions up than right for the right boundary
    self.generate_track_boundaries(init_row, initial_stop_col, final_stop_row, final_col, 0.5)

    self.racetrack[init_row, initial_start_col:initial_stop_col+1] = 1 # Setting all initial vehicle positions to '1'
    self.racetrack[final_start_row:final_stop_row+1, final_col] = 3 # Setting all final vehicle positions to '3'

    self.finalize_racetrack()
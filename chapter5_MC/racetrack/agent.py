import numpy as np

class Agent:

  def __init__(self, env, num_episodes, min_velocity, max_velocity):
    self.env = env
    self.num_episodes = num_episodes

    self.min_velocity = min_velocity # min vehicle velocity
    self.max_velocity = max_velocity # max vehicle velocity
    self.rest_velocity = (0, 0)
    
    self.vehicle_velocity = [(vx, vy) for vx in range(self.min_velocity, self.max_velocity+1) for vy in range(self.min_velocity, self.max_velocity+1)]
    
    # Checking if velocity matches the given constraint
    self.valid_velocity_constraint = lambda vel: np.all(np.array(vel) >= self.min_velocity) and np.all(np.array(vel) <= self.max_velocity)
    
    self.define_actions()
    self.get_all_valid_actions()

  def define_actions(self):
    '''
    Define all possible actions
    '''
    self.actions = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]

  def get_all_valid_actions(self):
    '''
    For each state, define all possible actions in a dictionary.
    '''
    self.all_valid_actions = {}

    for s in self.env.states:
      self.all_valid_actions[s] = []

      velocity = (s[2], s[3])
      for a in self.actions:
        new_velocity = tuple(np.add(velocity, a))
        if new_velocity != self.rest_velocity and self.valid_velocity_constraint(new_velocity):
          self.all_valid_actions[s].append(a)

      num_valid_actions = len(self.all_valid_actions[s])
      assert num_valid_actions > 0, print("State has no actions: ", s)

  def get_action_probabilities(self, state):
    '''
    For a given state and valid actions chosen, get action probabilities 
    for a epsilon-greedy behavior policy w.r.t current target policy.
    '''

    self.action_probs = []
    num_valid_actions = len(self.all_valid_actions[state])

    if self.policy[state] in self.all_valid_actions[state]:
      for va in self.all_valid_actions[state]:
        if va == self.policy[state]:
          p = 1 - self.epsilon + (self.epsilon/num_valid_actions)
        else:
          p = (self.epsilon/num_valid_actions)
        
        self.action_probs.append(p)

    else:
      self.action_probs = list(np.ones(num_valid_actions)*(1/num_valid_actions))
  
  def get_new_state(self):
    # Get new state
    select_initial_pos = np.random.choice(len(self.env.init_racetrack_positions))
    return self.env.init_racetrack_positions[select_initial_pos], self.rest_velocity
  
  def generate_episode(self, epsilon, policy, evaluate_target_policy=False):
    '''
    Generate episode for a behavior/target policy.
    '''

    self.epsilon = epsilon
    self.policy = policy
    
    episode = {'S':[], 'A':[], 'R':[], 'b':[], 'ep_len': 0}
    vehicle_pos, velocity = self.get_new_state()

    while vehicle_pos not in self.env.final_racetrack_positions:
      
      s_t = vehicle_pos + velocity
      episode['S'].append(s_t)

      if not evaluate_target_policy:
        self.get_action_probabilities(s_t)
        select_action = np.random.choice(len(self.all_valid_actions[s_t]), p=self.action_probs) # Random behavior policy
        action_for_s = self.all_valid_actions[s_t][select_action]
        episode['b'].append(self.action_probs[select_action]) # Append b(A_t|S_t) probability
      elif evaluate_target_policy:
        # Target policy
        action_for_s = self.policy[s_t]

      episode['A'].append(action_for_s)
      
      # Update position and velocity
      velocity = tuple(np.add(velocity, action_for_s))
      vehicle_pos = (vehicle_pos[0] - velocity[0], vehicle_pos[1] + velocity[1])

      if vehicle_pos not in self.env.racetrack_positions:
        episode['R'].append(min(self.env.reward))
        vehicle_pos, velocity = self.get_new_state()
        if evaluate_target_policy:
          episode['S'].append(vehicle_pos + velocity)
          episode['ep_len'] = len(episode['S'])
          return episode, False
      else:
        episode['R'].append(max(self.env.reward))

      # If vehicle has reached any of the final position      
      if vehicle_pos in self.env.final_racetrack_positions:
        episode['S'].append(vehicle_pos + self.rest_velocity)
        episode['ep_len'] = len(episode['S'])
        if evaluate_target_policy:
          return episode, True
    
    return episode, None
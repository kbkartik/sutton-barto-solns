import numpy as np
from tqdm import tqdm

class MC:
  def __init__(self, env, agent, gamma, epsilon, weighted_is=True, seed=374):
    self.env = env
    self.agent = agent
    self.gamma = gamma
    self.epsilon = epsilon
    self.weighted_is = weighted_is # Determines whether we are using weighted or ordinary IS.
    self.set_seed(seed)
    #self.reset()

  def set_seed(self, seed):
    np.random.seed(seed)

  def reset(self):
    raise NotImplementedError



class OffPolicyMCC(MC):
  def __init__(self, env, agent, gamma, epsilon):
    super().__init__(env, agent, gamma, epsilon)

    # Initialize q values with negative numbers for faster convergence.
    self.Q = {(s, a): np.random.randint(-200, 0) for s in self.env.states for a in self.agent.actions}
    self.C = {(s, a): 0 for s in self.env.states for a in self.agent.actions} # Initialize count to 0.
    self.init_target_policy()
    self.set_of_episodes = []
  
  def init_target_policy(self):
    '''
    Initialize a random deterministic greedy target policy.
    '''

    self.target_policy = {}
    for s in self.env.states:
      self.update_target_policy(s)
  
  def update_target_policy(self, s):
    '''
    Get q values for a particular state, choose the best action, 
    and update target policy.
    '''
    
    #q_s_a = [self.Q[(s,a)] for a in self.agent.all_valid_actions[s] if (s, a) in self.Q]
    q_s_a = [self.Q[(s,a)] for a in self.agent.actions if (s, a) in self.Q]
    
    assert len(q_s_a) > 0, print("Issue in updating target policy: ", s)

    #best_action = self.agent.all_valid_actions[s][np.argmax(q_s_a)]
    best_action = self.agent.actions[np.argmax(q_s_a)]
    self.target_policy[s] = best_action
    
  def run(self):

    # Run off-policy MC control for a specific number of episodes
    for i in tqdm(range(self.agent.num_episodes)):

      # Generating episode using behavior policy w.r.t target policy
      episode, _ = self.agent.generate_episode(self.epsilon, self.target_policy)
      self.set_of_episodes.append(episode)

      G = 0
      W = 1
      T = episode['ep_len'] - 1
      for t in reversed(range(T)):

        # In this code, we follow the episode convention S_0, A_0, R_0, S_1, A_1, R_1, S_2, ...., R_N
        # rather than following Sutton & Barto's convention S_0, A_0, R_1, S_1, A_1, R_2, S_2, ...., R_N.
        S_t = episode['S'][t]
        A_t = episode['A'][t]

        G = self.gamma * G + episode['R'][t]
        self.C[(S_t, A_t)] += W
        self.Q[(S_t, A_t)] += (W/self.C[(S_t, A_t)]) * (G - self.Q[(S_t, A_t)])
        
        self.update_target_policy(S_t)

        if A_t != self.target_policy[S_t]:
          break

        # Assuming a deterministic greedy policy, we divide W by b(A_t|S_t) as per the algorithm.
        W /= episode['b'][t]
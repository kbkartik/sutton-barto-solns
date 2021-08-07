import numpy as np
from environment import Environment
from agent import Agent
from mc import OffPolicyMCC
from visualizer import Visualizer

rows = 40
cols = 15
min_vehicle_velocity = 0
max_vehicle_velocity = 4
num_episodes = 100000
discount_factor = 1
epsilon = 0.2

env = Environment(rows, cols, min_vehicle_velocity, max_vehicle_velocity, 'racetrack.npy')
agent = Agent(env, num_episodes, min_vehicle_velocity, max_vehicle_velocity)

off_policy_mcc = OffPolicyMCC(env, agent, discount_factor, epsilon)
off_policy_mcc.run()
viz = Visualizer(rows, cols, off_policy_mcc)

# Evaluating (possibly) converged target policy
target_episode, target_policy_success = agent.generate_episode(off_policy_mcc.epsilon, off_policy_mcc.target_policy, evaluate_target_policy=True)

if target_policy_success: 
  print('Vehicle reached final position') 
  episode_in_question = target_episode
  for s, i in zip(episode_in_question['S'], range(episode_in_question['ep_len'])):
    pos = (s[0], s[1])
    if i == episode_in_question['ep_len'] - 1:
      viz.visualize_racetrack(pos, terminal=True)
    else:
      viz.visualize_racetrack(pos)
  viz.close_window(7, '/content/video.gif')
else:
  print('Vehicle could not reached final position')
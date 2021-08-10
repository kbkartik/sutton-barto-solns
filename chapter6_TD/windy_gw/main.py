import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
from sarsa import Sarsa

def plot_figure(filename, time_steps, color, label):
  plt.figure(figsize=(8, 8), dpi=70)
  plt.plot(range(len(time_steps)), time_steps, color=color, label=label)
  plt.legend()
  plt.xlabel('Num of episodes')
  plt.ylabel('Num of steps')
  plt.savefig(filename)

# 4 actions
agent = Agent(0.5, 0.1, 1)
env = Environment(7, 10, [0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
sarsa = Sarsa(agent, env, 200)
four_actions_time_steps = sarsa.run()
plot_figure('ex_6_9_four_actions.png', four_actions_time_steps, 'r', '4-actions')

# 8 actions
agent = Agent(0.5, 0.1, 1, diagonal=True)
sarsa = Sarsa(agent, env, 200)
eight_actions_time_steps = sarsa.run()
plot_figure('ex_6_9_eight_actions.png', eight_actions_time_steps, 'b', '8-actions')

# 9 actions
agent = Agent(0.5, 0.1, 1, diagonal=True, ninth_action=True)
sarsa = Sarsa(agent, env, 200)
nine_actions_time_steps = sarsa.run()
plot_figure('ex_6_9_nine_actions.png', nine_actions_time_steps, 'k', '9-actions')

# stochastic wind with 8-actions
agent = Agent(0.5, 0.1, 1, diagonal=True)
env = Environment(7, 10, [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], stochastic_wind=True)
sarsa = Sarsa(agent, env, 400)
stochastic_wind_time_steps = sarsa.run()
plot_figure('ex_6_10_stochastic_wind.png', stochastic_wind_time_steps, 'g', 'stochastic-wind')
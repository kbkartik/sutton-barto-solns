# Adapted from https://gist.github.com/thunderInfy/18931ffd6242c56cf386bdd31f78e61b

import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from google.colab import output
import time
import imageio
import os, sys
import pygame

# set SDL to use the dummy NULL video driver, 
# so it doesn't need a windowing system.
# This setting is done in Google Colab.
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()

class Visualizer:
    
  #HELPFUL FUNCTIONS
  def create_window(self):
    '''
    Creates window and assigns self.display variable
    '''
    self.display = pygame.display.set_mode((self.width, self.height))
    pygame.display.set_caption("Racetrack")
  
  def setup(self):
    '''
    Does things which occur only at the beginning
    '''
    self.cell_edge = 9
    self.width = 20*self.cell_edge
    self.height = 40*self.cell_edge
    self.create_window()
    self.window = True
    
  def close_window(self, fps, savepath):
    self.window = False
    imageio.mimsave(savepath, self.ep_gifs, fps=fps)

    pygame.quit()

  def draw(self, terminal, state = np.array([])):
    self.display.fill(0)
    for i in range(self.rows):
      for j in range(self.cols):
        if self.offmcc.env.racetrack[i,j] != 0:
          if self.offmcc.env.racetrack[i,j] == 2:
              color = (255,0,0)
          elif self.offmcc.env.racetrack[i,j] == 1:
              color = (255,255,0)
          elif self.offmcc.env.racetrack[i,j] == 3:
              color = (0,255,0)
          pygame.draw.rect(self.display,color,((j*self.cell_edge,i*self.cell_edge),(self.cell_edge,self.cell_edge)),1)
    
    if len(state)>0:
      if not terminal:
        pygame.draw.rect(self.display,(255,255,255),((state[1]*self.cell_edge,state[0]*self.cell_edge),(self.cell_edge,self.cell_edge)),0)
      else:
        pygame.draw.rect(self.display,(0,0,255),((state[1]*self.cell_edge,state[0]*self.cell_edge),(self.cell_edge,self.cell_edge)),0)
    
    #convert image so it can be displayed in OpenCV
    view = pygame.surfarray.array3d(self.display)

    #  convert from (width, height, channel) to (height, width, channel)
    view = view.transpose([1, 0, 2])

    #  convert from rgb to bgr
    img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    self.ep_gifs.append(img_bgr)
    
    #Display image, clear cell every 0.5 seconds
    #cv2_imshow(img_bgr)
    #time.sleep(5)
    #output.clear()
    self.loop = False
            
    return None
      
  def visualize_racetrack(self, state = np.array([]), terminal=False):
    '''
    Draws Racetrack in a pygame window
    '''
    if self.window == False:
      self.setup()
    self.loop = True
    while(self.loop):
        ret = self.draw(terminal, state)
        if ret!=None:
            return ret
  
  #CONSTRUCTOR
  def __init__(self, rows, cols, offmcc):
    self.rows = rows
    self.cols = cols
    self.ep_gifs = []
    self.offmcc = offmcc
    self.window = False
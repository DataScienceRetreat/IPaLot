# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:26:37 2018

@author: Orlando Ciricosta
"""

import pygame
from A3C import Brain, Optimizer, Environment
from cfg import OPTIMIZERS, WIDTH, HEIGHT, ENVIRONMENTS
import time


def main():
    
# Initialize screen
    size = WIDTH, HEIGHT
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Basic parking traing')
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    color = (50, 50, 50)
    background.fill(color)
    screen.blit(background, (0, 0))
    pygame.display.flip()

# Initialise brain, optimizer threads, and environment threads
    brain = Brain()
    optimizers = [Optimizer(brain) for i in range(OPTIMIZERS)]
    environments = [Environment(brain) for i in range(ENVIRONMENTS)]

    #render for env[0]
    environments[0].render = True
    environments[0].screen = screen

    for o in optimizers:
        o.start()
        
    for e in environments:
        e.start()

# Then train for a fixed time
    time.sleep(10)
        
    for o in optimizers:
        o.stop()
    for o in optimizers:
        o.join()

    for e in environments:
        e.stop()
    for e in environments:
        e.join()

    
if __name__ == '__main__': main()

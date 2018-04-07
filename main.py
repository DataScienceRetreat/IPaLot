# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:26:37 2018
@author: Orlando Ciricosta

IPaLot: an Intelligent Parking Lot
This code is a training simulation for an AI-driven parking lot, based on
reinforcement learning. The idea is to train the system to take control of
the customer's car and dispatch/retrieve it from a designate parking spot.

The main creates a pygame environment where cars need to reach designated
parking spots within a lot, using 3 kind of objects:
1 - Brain() -- based on the A3C algorithm
2 - Optimizer() threads -- updating the brain according to experience
3 - Environment() threads -- playing game episodes and accumulating experience

After a fixed running time the program will stop training and return

"""

import pygame
from A3C import Brain, Optimizer, Environment
from cfg import OPTIMIZERS, WIDTH, HEIGHT, ENVIRONMENTS, TRAINING_TIME
import time, sys


def main():
    
# Initialize screen
    size = WIDTH, HEIGHT
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Basic parking training')
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    color = (50, 50, 50)
    background.fill(color)
    screen.blit(background, (0, 0))
    pygame.display.flip()
    pygame.font.init()

# Initialise brain, optimizer threads, and environment threads
#    brain = Brain(load_weights=True)
    brain = Brain()
    optimizers = [Optimizer(brain) for i in range(OPTIMIZERS)]
    environments = [Environment(brain) for i in range(ENVIRONMENTS)]
    sys.stdout = open('rewards.txt', 'w')

    #render for env[0] -- comment this lines out for training overnight
    environments[0].render = True
    environments[0].screen = screen
    
# get 1 greedy/deterministic env and 1 completely random    
    environments[0].eps_start = 0 # render a deterministic policy environment
    environments[0].eps_end = 0
    if ENVIRONMENTS > 2:
        environments[1].eps_start = 1 
        environments[1].eps_end = 1   
    environments[0].train = False # do not learn from the greedy bastard
    
    # write down episode rewards only for env[0], the deterministic one
    environments[0].verbose = True

    for o in optimizers:
        o.start()
        
    for e in environments:
        e.start()

# Then train for a fixed time
    time.sleep(TRAINING_TIME)
        
    for o in optimizers:
        o.stop()
    for o in optimizers:
        o.join()

    for e in environments:
        e.stop()
    for e in environments:
        e.join()

    
if __name__ == '__main__': main()

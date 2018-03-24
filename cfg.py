# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:30:44 2018
@author: Orlando Ciricosta


"""

# number of moving cars per episode
N_CARS = 1

# number of threads
OPTIMIZERS = 2
ENVIRONMENTS = 4

#-- constants for A3C
import numpy as np
INPUT_SHAPE = (None, 29, 4, 1)
NONE_STATE = [np.zeros((4,)), np.zeros((4,)), np.zeros((29,4,1))]
'''shape for the colliding cars input, there are 29 of them,
each with a rect (4 numbers). The Filled_Lot class in Group_handler.py
is where the total number of cars 29+1 comes from'''

NUM_ACTIONS=7

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

GAMMA = 0.99

N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = 0.1
EPS_STEPS = 4e6

THREAD_DELAY = 0.001

MAX_FRAMES = 10000

# bkg color
BACKGROUND_COLOR = (50, 50, 50)

#screen size
WIDTH, HEIGHT = 700, 400
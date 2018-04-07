# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:30:44 2018
@author: Orlando Ciricosta


"""

# training time in seconds
TRAINING_TIME = 3600*8

# number of moving cars per episode
N_CARS = 4

# number of threads
OPTIMIZERS = 3
ENVIRONMENTS = 8

#-- constants for A3C
import numpy as np
INPUT_SHAPE = (None, 29, 4, 1)
NONE_STATE = [np.zeros((4,)), np.zeros((4,)), np.zeros((29,4,1))]
'''shape for the colliding cars input, there are 29 of them,
each with a rect (4 numbers). The Filled_Lot class in Group_handler.py
is where the total number of cars 29+1 comes from'''

COLLISION_PENALTY = 0.2

NUM_ACTIONS=7 # do not change

MIN_BATCH = 128
LEARNING_RATE = 5e-4
RMSP_DECAY = 0.9

LOSS_V = 0.5			# v loss coefficient
LOSS_ENTROPY = 0.1 	# entropy coefficient

GAMMA = 0.999

N_STEP_RETURN = 10
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.8
EPS_STOP  = 0.5
EPS_STEPS = 5e6

THREAD_DELAY = 0.001

MAX_FRAMES = 10000

# Epochs between saving the weights
EPOCHS_PER_SAVE = 1000 

# bkg color
BACKGROUND_COLOR = (50, 50, 50)

#screen size
WIDTH, HEIGHT = 700, 400
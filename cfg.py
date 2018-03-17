# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:30:44 2018
@author: Orlando Ciricosta


"""

# number of moving cars per episode
N_CARS = 4

# constants for car physics
MAXSPEED = 0.4
MINSPEED = -0.4
MAXSTEERING = 35 #degrees
STEERING = 0.5
ACCELERATION = 0.08
SOFTENING = 0.04
STEER_SOFTENING = 0.2

# initial car positions heigth
FROM_BOTTOM = 45

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

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN


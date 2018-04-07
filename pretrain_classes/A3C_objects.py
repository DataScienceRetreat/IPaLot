# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 17:36:33 2018
@author: Orlando Ciricosta

class Brain(): modified version of the brain in A3C.py, with the same NN
    architecture, but a modified fit functionality: this uses Keras to pretrain
    the model according to the A* results

memory_push(states, actions, rewards, states_, terminal_s_,
                 stop_training, memories, tot_rewards, brain):
    same as the memory_train member of Environment() in A3C.py
    
get_current_reward(manager): same as the homonymous member of Agent() in A3C.py

"""
import numpy as np
from cfg import GAMMA, GAMMA_N, INPUT_SHAPE, NONE_STATE, NUM_ACTIONS

import keras.models as models
import keras.layers as layers
from keras.layers.advanced_activations import PReLU

FITTERS = 3
MIN_BATCH = 128

#------------------------------------------------------------------------------
class Brain():
    def __init__(self, load_weights=False):
        self.train_queue = [ [], [], [], [], [] ]
        # s, a, r, s', s' terminal mask
        
        self.load_weights=load_weights
        self.model = self._build_model()

    def _build_model(self):
        ''' The model has 3 inputs as defined in the get_states method
        of the Car_handler class (in Group_handler.py)'''

        # driving car input branch
        player = layers.Input( batch_shape=(None, 4) )
#        dense1 = layers.Dense(8, activation = 'relu')(player)
        dense1 = PReLU()(layers.Dense(
                            8, kernel_initializer='random_uniform',
                            bias_initializer='random_uniform'
                            )(player)
                        )
        
        # target position input branch
        target = layers.Input( batch_shape=(None, 4) )
#        dense2 = layers.Dense(8, activation = 'relu')(target)
        dense2 = PReLU()(layers.Dense(
                            8, kernel_initializer='random_uniform',
                            bias_initializer='random_uniform'
                            )(target)
                        )
        
        # objects-to-avoid input branch
        mov_sta = layers.Input( batch_shape=INPUT_SHAPE )
#        conv = layers.Conv2D(8, (1,4), activation="relu")(mov_sta)
        conv = PReLU()(layers.Conv2D(
                            8, (1,4), kernel_initializer='random_uniform',
                            bias_initializer='random_uniform'
                            )(mov_sta)
                        )
        flat = layers.Flatten()(conv)
#        dense5 = layers.Dense(8, activation = 'relu')(flat)
        dense5 = PReLU()(layers.Dense(
                            8, kernel_initializer='random_uniform',
                            bias_initializer='random_uniform'
                            )(flat)
                        )
        
        # merge the first 2 branches 
        conc1 = layers.concatenate([dense1,dense2])
#        dense3 = layers.Dense(4, activation='relu')(conc1)
        dense3 = PReLU()(layers.Dense(
                            4, kernel_initializer='random_uniform',
                            bias_initializer='random_uniform'
                            )(conc1)
                        )
        
        # then merge with the third
        conc2 = layers.concatenate([dense3, dense5])
#        dense4 = layers.Dense(16, activation='relu')(conc2)
        dense4 = PReLU()(layers.Dense(
                            16, kernel_initializer='random_uniform',
                            bias_initializer='random_uniform'
                            )(conc2)
                        )
        
        # finally split into the 2 outputs (policy,value)
        policy = layers.Dense(NUM_ACTIONS, activation='softmax',
                              kernel_initializer='random_uniform',
                              bias_initializer='random_uniform')(dense4)
        value   = layers.Dense(1, activation='linear', 
                               kernel_initializer='random_uniform',
                               bias_initializer='random_uniform')(dense4)

        model = models.Model(
                inputs=[player, target, mov_sta],
                outputs=[policy, value]
                )
        
        model.compile(
        optimizer = "rmsprop",
        loss={'dense_7': 'mean_squared_error',
              'dense_6': 'categorical_crossentropy'},
        metrics = {'dense_7': 'mse',
              'dense_6': 'accuracy'},
        )
        
        if self.load_weights:
            model.load_weights('weights/pretrain_weights.h5')
        
        model.summary()

        return model

    def train_push(self, s, a, r, s_):
        self.train_queue[0].append(s)
        self.train_queue[1].append(a)
        self.train_queue[2].append(r)

        if s_ is None:
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
        else:
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)
            
    def fit(self):
        
        if len(self.train_queue[0]) < MIN_BATCH:
            return 

        s, a, r, s_, s_mask = self.train_queue
        self.train_queue = [ [], [], [], [], [] ]

        # stack the 3 input parts separately for training
        s1 = np.stack(np.stack(s)[:,0]) 
        s2 = np.stack(np.stack(s)[:,1])
        s3 = np.stack(np.stack(s)[:,2])
       
        a = np.vstack(a) # target policy
        r = np.vstack(r) # target value
        

        x_train = [s1, s2, s3]
        y_train = [a, r]
        
        self.model.fit(
                x_train, y_train,
                epochs=50, batch_size=MIN_BATCH
            )

        

#------------------------------------------------------------------------------
def get_current_reward(manager):
    '''calculate the reward r based on distance, using a linear potential.
        The potential increases from 0 to 1 as the car gets closer to the
        target; r is such that when updating the total reward R+=r then R
        will be the potential in the newly reached position'''

    reward = 0
    car = manager.moving_cars[0]
        
    # get current distances player-target
    fwp = car.get_frontwheel(negative_y = False)
    rwp = car.get_rearwheel(negative_y = False)
    fwt = manager.current_target[0][0]
    rwt = manager.current_target[0][1]
    dist0= manager.get_distance(fwp,fwt) 
    dist1= manager.get_distance(rwp,rwt, drawpath=True)
    
    # retrieve constants for the potential
    c0 = manager.const[0][0]
    c1 = manager.const[0][1]

    if manager.last_distances[0]:
    # if this is not the first step after acquiring a new target
    
        # update front wheel potential and distance
        previous_R = potential(manager.last_distances[0][0], c0)
        updated_R = potential(dist0, c0)
        reward += updated_R - previous_R
            # it will be negative if the car is moving away from the target
#        manager.last_distances[0][0] = dist0        
        
        # repeat for rear wheel
        previous_R = potential(manager.last_distances[0][1], c1)
        updated_R = potential(dist1, c1)
        reward += updated_R - previous_R
#        manager.last_distances[0][1] = dist1  

    else:
    # if a new target has just been set just save the distances
    
        manager.last_distances[0].append(dist0)
        manager.last_distances[0].append(dist1)
        manager.const[0][0] = dist0
        manager.const[0][1] = dist1

    return reward

#------------------------------------------------------------------------------
        
def potential(r, const):
    '''Auxiliary function that returns the distance based linear
    potential for reward engineering: it linearly increases
    from 0 to 1/2 between r=const and r=0, so that when both the 
    peusowheels are on target the total potential is 1/2 + 1/2 = 1 '''

    return (const-r)/(2*const)

#------------------------------------------------------------------------------
    
def memory_push(states, actions, rewards, states_, terminal_s_,
                 stop_training, memories, tot_rewards, brain):
    N_CARS = 1
    for i in range(N_CARS):
        if not stop_training[i]:
            # first push s,a,r,s' into individual memory
            if not terminal_s_[i]:
                memories[i].append( [ states[i],
                                    actions[i],
                                    rewards[i],
                                    states_[i] ] )
            else:
                memories[i].append( [ states[i],
                                    actions[i],
                                    rewards[i],
                                    None ] )    
    
            tot_rewards[i] = ( tot_rewards[i] +
                                    rewards[i] * GAMMA_N ) / GAMMA

            # send memory to the training queue if s' is terminal
            if terminal_s_[i]:
                while len(memories[i]) > 0:
                    n = len(memories[i])
                    s, a, _, _  = memories[i][0]
                    _, _, _, s_ = memories[i][n-1]
                    r = tot_rewards[i]
                    brain.train_push(s, a, r, s_)

                    tot_rewards[i] = ( tot_rewards[i]
                                - memories[i][0][2] ) / GAMMA
                    memories[i].pop(0)		

                tot_rewards[i] = 0

            # Only use memory from terminal states: removed N-steps update
                

        

        

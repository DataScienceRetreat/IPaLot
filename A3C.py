# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:39:27 2018
@author: Orlando Ciricosta

Based on the A3C implementation by Jaromir Janisch, 2017
available under MIT license at
https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

"""

import numpy as np
import tensorflow as tf

import time, random, threading

import keras.models as models
import keras.layers as layers
from keras import backend as K

#-- constants

INPUT_SHAPE = (None, 29, 4, 1)
NONE_STATE = [np.zeros((4,)), np.zeros((4,)), np.zeros((29,4,1))]
'''shape for the colliding cars input, there are 29 of them,
each with a rect (4 numbers). The Filled_Lot class in Group_handler.py
is where the total number of cars 29+1 comes from'''

NUM_ACTIONS=7

RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

#---------
class Brain():
    def __init__(self):
        self.train_queue = [ [], [], [], [], [] ]
        # s, a, r, s', s' terminal mask
        
        self.lock_queue = threading.Lock()        
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()	# avoid modifications

    def _build_model(self):
        ''' The model has 3 inputs as defined in the get_states method
        of the Car_handler class (in Group_handler.py)'''

        # driving car input branch
        player = layers.Input( batch_shape=(None, 4) )
        dense1 = layers.Dense(8, activation = 'relu')(player)
        
        # target position input branch
        target = layers.Input( batch_shape=(None, 4) )
        dense2 = layers.Dense(8, activation = 'relu')(target)
        
        # objects-to-avoid input branch
        mov_sta = layers.Input( batch_shape=INPUT_SHAPE )
        conv = layers.Conv2D(8, (1,4), activation="relu")(mov_sta)
        flat = layers.Flatten()(conv)
        
        # merge the first 2 branches 
        conc1 = layers.concatenate([dense1,dense2])
        dense3 = layers.Dense(4, activation='relu')(conc1)
        
        # then merge with the third
        conc2 = layers.concatenate([dense3, flat])
        dense4 = layers.Dense(16, activation='relu')(conc2)
        
        # finally split into the 2 outputs (policy,value)
        policy = layers.Dense(NUM_ACTIONS, activation='softmax')(dense4)
        value   = layers.Dense(1, activation='linear')(dense4)

        model = models.Model(
                inputs=[player, target, mov_sta],
                outputs=[policy, value]
                )
        model._make_predict_function()	# have to initialize before threading
        
        model.summary()

        return model

    def _build_graph(self, model):
        s1_t = tf.placeholder(tf.float32, shape=(None, 4))
        s2_t = tf.placeholder(tf.float32, shape=(None, 4))        
        s3_t = tf.placeholder(tf.float32, shape=INPUT_SHAPE)        
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))
        # not immediate, but discounted n step reward

        p, v = model([s1_t, s2_t, s3_t])

        log_prob = tf.log(
                tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10
                )
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
        loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(
                p * tf.log(p + 1e-10), axis=1, keep_dims=True
                )	# maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s1_t, s2_t, s3_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)	# yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:	# more thread could have passed without lock
                return 									# we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5*MIN_BATCH:
            print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state

        s1_t, s2_t, s3_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize,
                feed_dict={s1_t: s[0], s2_t: s[1], s3_t: s[2], a_t: a, r_t: r}
                )

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

        if s_ is None:
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
        else:
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v
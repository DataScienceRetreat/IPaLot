# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:39:27 2018
@author: Orlando Ciricosta

Based on the A3C implementation by Jaromir Janisch, 2017
available under MIT license at
https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py


class Brain():
    sets up the NN predicting policy/value for A3C
    saves the weights every EPOCHS_PER_SAVE epochs

class Optimizer{threading.Thread}(brain):
    sets up optimizer threads executing brain.optimize() for the global brain

class Environment{threading.Thread}(brain, render_on=None, eps_start=EPS_START,
                 eps_end=EPS_STOP, eps_steps=EPS_STEPS, verbose=False):
    each environment will run episodes with N_CARS agents, update the state
    at each frame, get the N_CARS rewards, and push the experience on the
    training queue
    
class Agent{threading.Thread}(brain,manager,eps_start,eps_end,eps_steps,
                 state,action_list,index,reward_list):
    each agent will compute a[i] = pi(s[i]), act on car[i],
    and compute a temporary reward based on distance, neglecting collisions

"""

import numpy as np
import pygame, random
import tensorflow as tf
import time, threading
import keras.models as models
import keras.layers as layers
from keras import backend as K
from cars.Group_handler import Car_handler
from keras.layers.advanced_activations import PReLU

from cfg import INPUT_SHAPE, NONE_STATE, NUM_ACTIONS, MIN_BATCH, LEARNING_RATE
from cfg import LOSS_V, LOSS_ENTROPY, GAMMA_N, EPS_START, EPS_STOP, EPS_STEPS
from cfg import THREAD_DELAY, N_CARS, BACKGROUND_COLOR, WIDTH, HEIGHT, GAMMA
from cfg import N_STEP_RETURN, MAX_FRAMES, EPOCHS_PER_SAVE, COLLISION_PENALTY
from cfg import RMSP_DECAY

#------------------------------------------------------------------------------
class Brain():
    def __init__(self, load_weights=False):
        self.train_queue = [ [], [], [], [], [] ]
        # s, a, r, s', s' terminal mask
        
        self.counter = 0 # update every time it trains,
                         # used to decrese epsilon, and saving weights
        self.lock_queue = threading.Lock()        
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        
        self.load_weights=load_weights

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        if self.load_weights:
            self.model.load_weights('weights/pretrain_weights.h5')
#            model.load_weights('weights/brain_weights.h5') 
        
        self.default_graph = tf.get_default_graph()

#        self.default_graph.finalize()	# avoid modifications

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

        loss_policy = - log_prob * tf.stop_gradient(advantage) #maximize policy
        loss_value  = LOSS_V * tf.square(advantage) # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(
                p * tf.log(p + 1e-10), axis=1, keep_dims=True
                )	# maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=RMSP_DECAY)
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

        # stack the 3 input parts separately for training
        s1 = np.stack(np.stack(s)[:,0]) 
        s2 = np.stack(np.stack(s)[:,1])
        s3 = np.stack(np.stack(s)[:,2])
        
        a = np.vstack(a)
        r = np.vstack(r)
        
        s1_ = np.stack(np.stack(s_)[:,0]) 
        s2_ = np.stack(np.stack(s_)[:,1])
        s3_ = np.stack(np.stack(s_)[:,2])        
        s_ = [s1_, s2_, s3_]

        s_mask = np.vstack(s_mask)

#        if len(s) > 5*MIN_BATCH:
#            print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state

        s1_t, s2_t, s3_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize,
                feed_dict={s1_t: s1, s2_t: s2, s3_t: s3, a_t: a, r_t: r}
                )
        with self.lock_queue:
            self.counter += 1
            if (self.counter % EPOCHS_PER_SAVE) == 0:
                self.model.save_weights('weights/brain_weights.h5')
                filename = 'weights/weights_' + str(
                        self.counter) + '_' + time.strftime(
                                "%m_%d_%H-%M-%S", time.gmtime()) + '.h5'
                self.model.save(filename)            
            # save twice, once on the file that keeps being overwritten, once
            # on a dated file (to retrieve if the system stops learning later)

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
        
        
#------------------------------------------------------------------------------
            
class Optimizer(threading.Thread):

    def __init__(self, brain):
        threading.Thread.__init__(self)
        self.brain = brain
        self.stop_signal = False

    def run(self):
        while not self.stop_signal:
            try:
                self.brain.optimize()
            except:
                print("# Warning: optimizer failed")
            # do not screw up a night worth of training if it fails once
                
    def stop(self):
        self.stop_signal = True
        
#------------------------------------------------------------------------------
        
class Environment(threading.Thread):

    def __init__(self, brain, render_on=None, eps_start=EPS_START,
                 eps_end=EPS_STOP, eps_steps=EPS_STEPS, verbose=False):
        
        threading.Thread.__init__(self)
        
        self.brain = brain
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        
        self.render = False
        self.screen = None

        self.stop_signal = False
        if render_on is not None:
            self.render = True
            self.screen = render_on
        
        self.stop_training = [] # a flag list: the i-th flag will get activated
                                # when car[i] is done, to stop sending s,a,r,s'
                                # to memory in the next frame
                                
        self.lock_queue = threading.Lock() # lock to access event queue

        self.train = True # allow to switch-off training is set to false
        self.verbose = verbose

    def runEpisode(self): 
        manager = Car_handler(N_CARS)

        terminal = [False]*N_CARS
        states = manager.get_states(terminal)
        self.stop_training = []
        self.tot_rewards = []
        manager.last_distances = []
        R = []
        self.memories = []

        for i in range(N_CARS):
            manager.last_distances.append([])
            self.memories.append([])
            self.tot_rewards.append(0)
            self.stop_training.append(False)
            R.append(0)
            
        frame = 0
        done = False 
        
        while True:
            with self.lock_queue:
                pygame.event.pump() # ensure pygame interacts correctly with OS

            time.sleep(THREAD_DELAY) # yield 
            
            agents = []
            actions = []
            one_hot_actions = []
            rewards = []
            terminal_flags = []
            
            for i in range(N_CARS):
                actions.append(None)
                one_hot_actions.append(np.zeros(NUM_ACTIONS))
                rewards.append(None)
                terminal_flags.append(False)
                agents.append(Agent( self.brain,
                                     manager,
                                     self.eps_start,
                                     self.eps_end,
                                     self.eps_steps,
                                     states[i],
                                     actions, i, # will write on action[i]
                                     rewards     # will write on rewards[i]
                                     )
                                )
        
                agents[i].start()
                # each agent will compute a[i] = pi(s[i]), act on car[i],
                # and compute a temporary reward based on distance, neglecting
                # collisions
                
            # now wait for all agents to do their job
            for agent in agents:
                agent.join()
                

            for i, car in enumerate(manager.moving_cars):                
                # check for collisions for each car that is not done
                if not manager.car_is_done[i]:
                    collision = True                
                    if car.rect.left < 0 or car.rect.right > WIDTH:
                        manager.reset_car(i)
                    elif car.rect.top < 0 or car.rect.bottom > HEIGHT:
                        manager.reset_car(i)                             
                    elif pygame.sprite.spritecollide(car,
                                               manager.static_cars_group,
                                               False,
                                               pygame.sprite.collide_mask):
                        manager.reset_car(i)                
                    elif pygame.sprite.spritecollide(car,
                                               manager.collide_with[i],
                                               False,
                                               pygame.sprite.collide_mask):            
                        manager.reset_car(i)
                    else:
                        collision = False
                        
                    if collision:
                        rewards[i] = COLLISION_PENALTY
                        # penalize collisions
                        
                        terminal_flags[i] = True
                        done = True # stop the episode even for 1 collision
                
                    
                # get one-hot enconding of the action
                one_hot_actions[i][actions[i]] = 1 

            # get the new state, the manager will also flag terminal states
            new_states = manager.get_states( terminal_flags )

            # push s,a,r,s' into each car's memory, and communicate to brain
            if self.train:
                self.memory_train(states, one_hot_actions,
                              rewards, new_states, terminal_flags)

            # if the get_states call gives car_is_done[i]=True for the 1st time
            # then the next calls to self.memory_train will ignore car i
            if manager.car_is_done[i] and not self.stop_training[i]:
                self.stop_training[i]=True
                
            states = new_states
            
            for i in range(N_CARS):
                if not terminal_flags[i]:
                # so that the updated R will show potential before a collision
                    R[i] += rewards[i]
            
            if self.render and frame%10 == 0:
                background = pygame.Surface(self.screen.get_size())
                background = background.convert()
                background.fill(BACKGROUND_COLOR)
                self.screen.blit(background, (0, 0))
                manager.moving_cars_group.draw(self.screen)
                manager.static_cars_group.draw(self.screen)
                pygame.draw.lines(self.screen,
                                  (0,0,0),
                                  False,
                                  manager.path_list,
                                  3)
#                self.render_reward(R[0]) # print the current reward for car[0]
                   
                pygame.display.flip()
            
            frame += 1
            
            #check if all cars are done if nothing has already triggered done
            if not done:
                done = True
                for i in range(N_CARS):
                    done = done and manager.car_is_done[i]
            
            if done or self.stop_signal or frame > MAX_FRAMES:
               break
        
        # print the episode reward averaged over cars
        if self.verbose:
            print(np.mean(R), flush=True)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True
        
        
    def memory_train(self, states, actions, rewards, states_, terminal_s_):

        for i in range(N_CARS):
            if not self.stop_training[i]:
                # first push s,a,r,s' into individual memory
                if not terminal_s_[i]:
                    self.memories[i].append( [ states[i],
                                        actions[i],
                                        rewards[i],
                                        states_[i] ] )
                else:
                    self.memories[i].append( [ states[i],
                                        actions[i],
                                        rewards[i],
                                        None ] )    
        
                self.tot_rewards[i] = ( self.tot_rewards[i] +
                                        rewards[i] * GAMMA_N ) / GAMMA
    
                # send memory to the training queue if s' is terminal
                if terminal_s_[i]:
                    while len(self.memories[i]) > 0:
                        n = len(self.memories[i])
                        s, a, _, _  = self.memories[i][0]
                        _, _, _, s_ = self.memories[i][n-1]
                        r = self.tot_rewards[i]
                        self.brain.train_push(s, a, r, s_)
    
                        self.tot_rewards[i] = ( self.tot_rewards[i]
                                    - self.memories[i][0][2] ) / GAMMA
                        self.memories[i].pop(0)		
    
                    self.tot_rewards[i] = 0
    
                # if enough steps have been accumulated in memory, send 
                # an N-STEPS result to the training queue
                if len(self.memories[i]) >= N_STEP_RETURN:
                    s, a, _, _  = self.memories[i][0]
                    _, _, _, s_ = self.memories[i][N_STEP_RETURN-1]
                    r = self.tot_rewards[i]
                    self.brain.train_push(s, a, r, s_)
    
                    self.tot_rewards[i] = ( self.tot_rewards[i]
                                                - self.memories[i][0][2] )
                    self.memories[i].pop(0)
                    
    def render_reward(self, number):
        font = pygame.font.Font(None, 36)
        text = '{:.4f}'.format(number)
        text_surface = font.render(text, True, (200, 155, 155))
        text_pos = text_surface.get_rect(centerx=self.screen.get_width()/2)
        self.screen.blit(text_surface, text_pos)


#------------------------------------------------------------------------------
           
class Agent(threading.Thread):

    def __init__(self,
                 brain,
                 manager,
                 eps_start,
                 eps_end,
                 eps_steps,
                 state,
                 action_list,
                 index,
                 reward_list
                 ):
        
        threading.Thread.__init__(self)
        
        self.brain = brain
        self.car = manager.moving_cars[index]
        self.manager = manager
        
        if(self.brain.counter >= eps_steps):
            self.epsilon = eps_end
        else:
            self.epsilon = eps_start + self.brain.counter * (eps_end - 
                        eps_start) / eps_steps	# linearly interpolate
        self.state = state
        self.action_list = action_list
        self.i = index
        self.reward_list = reward_list
        
    def run(self):
        '''each agent will compute a[i] = pi(s[i]), act on car[i],
        and compute a temporary reward based on distance, neglecting
        collisions'''
        
        # get new action according to epsilon-greedy policy
        if random.random() < self.epsilon:
            a = random.randint(0, NUM_ACTIONS-1)
        else:
            #transform state in a len-1 batch (needed by brain.predict)
            s = []
            for inpt in self.state:
                s.append(np.array( [ inpt ] ))
            #get policy 
            p = self.brain.predict_p(s)[0]
                # the [0] is once again just for shape reasons

            if self.epsilon > 1e-10:
                a = np.random.choice(NUM_ACTIONS, p=p)
            else:
            # make a greedy environment fully deterministic
                a = np.argmax(p)
        
        # act and save action in the list for external access
        self.car.act(a)
        self.action_list[self.i] = a
        
        # finally get the distance-based reward            
        self.reward_list[self.i] = self.get_current_reward()
        
            
    def get_current_reward(self):
        '''calculate the reward r based on distance, using a linear potential.
            The potential increases from 0 to 1 as the car gets closer to the
            target; r is such that when updating the total reward R+=r then R
            will be the potential in the newly reached position'''

        reward = 0
            
        # get current distances player-target
        fwp = self.car.get_frontwheel(negative_y = False)
        rwp = self.car.get_rearwheel(negative_y = False)
        fwt = self.manager.current_target[self.i][0]
        rwt = self.manager.current_target[self.i][1]
        dist0= self.manager.get_distance(fwp,fwt) 
        dist1= self.manager.get_distance(rwp,rwt, drawpath=(self.i == 0))
        
        # retrieve constants for the potential
        c0 = self.manager.const[self.i][0]
        c1 = self.manager.const[self.i][1]

        if self.manager.last_distances[self.i]:
        # if this is not the first step after acquiring a new target
        
            # update front wheel potential and distance
            previous_R = potential(self.manager.last_distances[self.i][0], c0)
            updated_R = potential(dist0, c0)
            reward += updated_R - previous_R
                # it will be negative if the car is moving away from the target
            self.manager.last_distances[self.i][0] = dist0        
            
            # repeat for rear wheel
            previous_R = potential(self.manager.last_distances[self.i][1], c1)
            updated_R = potential(dist1, c1)
            reward += updated_R - previous_R
            self.manager.last_distances[self.i][1] = dist1  

        else:
        # if a new target has just been set just save the distances
        
            self.manager.last_distances[self.i].append(dist0)
            self.manager.last_distances[self.i].append(dist1)
            self.manager.const[self.i][0] = dist0
            self.manager.const[self.i][1] = dist1

        return reward
    
    def stop(self):
        self.stop_signal = True
        
        
#------------------------------------------------------------------------------
        
def potential(r, const):
    '''Auxiliary function that returns the distance based linear
    potential for reward engineering: it linearly increases
    from 0 to 1/2 between r=const and r=0, so that when both the 
    peusowheels are on target the total potential is 1/2 + 1/2 = 1 '''

    return (const-r)/(2*const)
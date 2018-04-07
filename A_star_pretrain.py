# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:12:00 2018
@author: Orlando Ciricosta

Pretrain the A3C brain for IPaLot, using an A* search algorithm
to identify the path to the target spot: it generates a target 
policy(state) and value(state) that will be used as y_train(x_train)
to train the NN
"""

import pygame
from pretrain_classes.Group_handler import Car_handler
from cfg import WIDTH, HEIGHT, NUM_ACTIONS, BACKGROUND_COLOR
from pretrain_classes.A3C_objects import Brain, get_current_reward, memory_push
from pretrain_classes.A_star_objects import Priority_queue, Vertex
from pygame.locals import QUIT
import numpy as np

# each car will act several times per each step of the loop   
consecutive_actions = 20


def save_path(come_from, current, graph, brain, car, manager, screen):
    ''' save the s,a,r,s' to the brain memory
    and render the path found by A* '''
    path = [current]
    while current in come_from.keys():
        current = come_from[current]
        path.append(current)        

    current = path.pop()
    graph[current].update(car)
    terminal_flags = [False]
    states = manager.get_states(terminal_flags)
    memories = [[]]
    tot_rewards = [0]
    
    while path:
        current = path.pop()
        a = graph[current].parent_action
        one_hot_action = np.zeros(NUM_ACTIONS)
        one_hot_action[a] = 1.0

        
        for i in range(consecutive_actions):
            car.act(a)
            new_states = manager.get_states(terminal_flags)
            rewards = [ get_current_reward(manager) ]
            
            background = pygame.Surface(screen.get_size())
            background = background.convert()
            background.fill(BACKGROUND_COLOR)
            screen.blit(background, (0, 0))
            manager.moving_cars_group.draw(screen)
            manager.static_cars_group.draw(screen)
            pygame.display.flip()
            if (not path) and (i == consecutive_actions -1):
                terminal_flags = [True]
            
            memory_push(states, [one_hot_action,],
                         rewards, new_states, terminal_flags,
                         [False], memories, tot_rewards, brain)
            states = new_states


#------------------------------------------------------------------------------
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
#    pygame.font.init()

# Initialise brain
    brain = Brain()

# Perform a A* for a path to the parking spot from each of N_START
# positions to each of the possible parking spots, accumulating s,a,r,s'
# in memory at the end of each episode.
# Once all the possible A* searches are done train
# the brain (Note lenght-1 lists/loops in the following are the result
# of reusing code written for the multi-agent A3C case)

    N_SPOTS = 1 #18 # number of parking spots
    N_STARTS = 1 #5 # number of starting positions

    N_CARS = 1 # do not change this
    
    exit_signal = False
    
    for i_spot in range(N_SPOTS):
        for i_start in range(N_STARTS):
        
            # start an episode
            manager = Car_handler(N_CARS, i_spot, i_start, N_STARTS)
            car = manager.moving_cars[0]
                
            # do a search for each of the target positions
#            while manager.target_positions[0]:
            if manager.target_positions[0]:
                target = manager.target_positions[0].pop()
                if manager.current_target:
                    manager.current_target[0]= target
                else:
                    manager.current_target.append(target)
                terminal = [False]*N_CARS
                states = manager.get_states(terminal)
                stop_training = []
                tot_rewards = []
                R = []
                memories = []
        
                for i in range(N_CARS):
                    memories.append([])
                    tot_rewards.append(0)
                    stop_training.append(False)
                    R.append(0)
                
                # initialize A* search
                open_set = Priority_queue()
                closed_set = set()
                come_from = {}
                gscore = {}
                fscore = {}
                graph = {} # dict to store vertices -- graph[vertex_id]=vertex
                
                start = Vertex(states[0], car, None, None, None)
                graph[start.id] = start
                open_set.add_task(start.id)
                gscore[start.id] = 0
                fscore[start.id] = manager.get_distance(start.fw, target[0]) \
                                + manager.get_distance(start.rw, target[1])
                
                done = False
                found = False
                    
                # here starts the forward search algorithm ------------------------
                while open_set.list:
                    
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            done = True
                            exit_signal = True

                    current = open_set.pop_task()
                    current_vertex = graph[ current ]
                    current_vertex.update(car)
                    
                    # check if the current vertex is a target position
                    # and update target if an intermediate position is reached
                    fwp = car.get_frontwheel(negative_y = False)
                    rwp = car.get_rearwheel(negative_y = False)
                    fwt = manager.current_target[0][0]
                    rwt = manager.current_target[0][1]
                    dist0= manager.get_distance(fwp,fwt,drawpath=True) 
                    dist1= manager.get_distance(rwp,rwt)
                    if ( dist1 <= 5 ) and ( dist0 <= 5 ):
                        found = True
    
                    if found:
                        save_path(come_from, current, graph,
                                  brain, car, manager, screen)
                    if done or found:
                        break
                    
                    closed_set.add(current)
                    
                    # update last_distances now rather than in in the 
                    # get_current_reward function (as in main), because  
                    # here I will call that function N_ACTIONS times                
                    manager.last_distances[0][0] = dist0
                    manager.last_distances[0][1] = dist1
                    
                    # render now before exploring
                    background = pygame.Surface(screen.get_size())
                    background = background.convert()
                    background.fill(BACKGROUND_COLOR)
                    screen.blit(background, (0, 0))
                    manager.moving_cars_group.draw(screen)
                    manager.static_cars_group.draw(screen)
                    pygame.draw.lines(screen,
                                  (0,0,0),
                                  False,
                                  manager.path_list,
                                  3)
                    pygame.display.flip()
                    
                    # now check all children and add them to the
                    # PQ if they don't collide and have not been explored
                    for a in range(NUM_ACTIONS):
                        for _ in range(consecutive_actions): 
                            car.act(a)
                            
    #                    manager.moving_cars_group.draw(screen)
    #                    pygame.display.flip()
    #                    time.sleep(1)
                            
    #                    one_hot_action = np.zeros(NUM_ACTIONS)
    #                    one_hot_action[a] = 1.0
                        
                        collision = False       
                        if car.rect.left < 0 or car.rect.right > WIDTH:
                            collision = True
                        elif car.rect.top < 0 or car.rect.bottom > HEIGHT:
                            collision = True                             
                        elif pygame.sprite.spritecollide(car,
                                                   manager.static_cars_group,
                                                   False,
                                                   pygame.sprite.collide_mask):
                            collision = True                
                        elif pygame.sprite.spritecollide(car,
                                                   manager.collide_with[i],
                                                   False,
                                                   pygame.sprite.collide_mask):            
                            collision = True
    
                        
                        if not collision:
                            reward = get_current_reward(manager)
                            terminal = [False]*N_CARS
                            new_states = manager.get_states(terminal)
                            
                            new_vertex = Vertex(new_states[0],
                                                car,
                                                current_vertex,
                                                a,
                                                reward)
                            if new_vertex.id not in gscore.keys():
                                gscore[new_vertex.id] = 1e10
                                fscore[new_vertex.id] = 1e10


                            tentative_gscore = gscore[current] + \
                                            new_vertex.parent_distance()
                            update = False
                            if tentative_gscore < gscore[new_vertex.id]:
                                come_from[new_vertex.id] = current
                                
                                gscore[new_vertex.id] = tentative_gscore
                                
                                fscore[new_vertex.id] = tentative_gscore + \
                                manager.get_distance(new_vertex.fw,target[0]) \
                                + manager.get_distance(new_vertex.rw,target[1])
                                
                                update = True
   
                            if (new_vertex.id not in graph.keys()) or update:                        
                                open_set.add_task(new_vertex.id,
                                            priority = fscore[new_vertex.id])
                                graph[new_vertex.id] = new_vertex
                            
    
                        # reset the car after trying each action
                        current_vertex.update(car)
                        
                # end of A* search---------------------------------------------
    
                if exit_signal:
                    break
                
                if not found:
                    print("Warning: A* ended with no success.") 
                    print("    Try and reduce consecutive_actions ")
                
            if exit_signal:
                break 
        if exit_signal:
            break          
            

    brain.fit()


# Finally save the weights        
    brain.model.save_weights('weights/pretrain_weights.h5')

    
if __name__ == '__main__': main()

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:26:37 2018

@author: Orlando Ciricosta
"""

import pygame
from pygame.locals import QUIT
from Group_handler import Car_handler
from A3C import Brain
from cfg import N_CARS, NUM_ACTIONS
import numpy as np


def main():
    
# Initialize screen
    size = width, height = 700, 400
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Basic parking traing')
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    color = (50, 50, 50)
    background.fill(color)
    screen.blit(background, (0, 0))
    pygame.display.flip()

#Initialise brain
    brain = Brain()

# Initialize car objects
    manager = Car_handler(N_CARS)


# Event loop
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            
        # get 'state'
        states = manager.get_states()

        #test randomly moving cars
        for i, car in enumerate(manager.moving_cars): 
            
            #transform state in a len-1 batch (needed by brain.predict)
            s = []
            for inp in states[i]:
                s.append(np.array( [ inp ] ))
            
            #get policy 
            p = brain.predict_p(s)[0]
                            # the [0] is once again just for shape reasons
            
            action_index = np.random.choice(NUM_ACTIONS, p=p)

#            action_index = random.randint(0,NUM_ACTIONS-1)            
            car.act(action_index)
                
            if car.rect.left < 0 or car.rect.right > width:
                manager.reset_car(i)
            if car.rect.top < 0 or car.rect.bottom > height:
                manager.reset_car(i)                 
            
            if pygame.sprite.spritecollide(car,
                                           manager.static_cars_group,
                                           False,
                                           pygame.sprite.collide_mask):
                manager.reset_car(i)
                
            if pygame.sprite.spritecollide(car,
                                           manager.collide_with[i],
                                           False,
                                           pygame.sprite.collide_mask):            
                manager.reset_car(i)
    
            car.update()        
            
        # render       
        screen.blit(background, (0, 0))
        manager.moving_cars_group.draw(screen)
        manager.static_cars_group.draw(screen)
                   
        pygame.display.flip()
        


    
if __name__ == '__main__': main()

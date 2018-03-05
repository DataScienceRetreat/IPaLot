# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:54:37 2018
@author: orlando

this module generates a target parking spot at a random position,
and surrounds it with static cars

input:
    spritegroup = the sprite group of the static cars (to set up collisions)
                  it will be rendered in main
    car = the car object I am trying to park. We need this for its dimensions
output:
    x_rear,y_rear, x_front, y_front,  = target positions of the pseudo wheels
"""

import pygame
import random
from Cars import Static_car
import math

def populator(spritegroup, car ):
    
    #pick random orientation of spot
    Vertical = random.choice([True,False])
    
    screenw = pygame.display.Info().current_w
    screenh = pygame.display.Info().current_h
    
    if Vertical:
        carw = car.rect.w
        carh = car.rect.h       
    else:
        carw = car.rect.h
        carh = car.rect.w
    
    #generate random position of the spot
    x = random.randint( 3*carw, screenw - 3*carw )
    y = random.randint( 3*carh, screenh - 3*carh )
    
    #get position of neighbour cars and add to sprite group
    align_y = random.choice([True,False])
    if align_y:
        position1 = ( x , y + math.floor(1.2*carh) )
        position2 = ( x , y - math.floor(1.2*carh) )
    else:
        position1 = ( x + math.floor(1.2*carw), y  )
        position2 = ( x - math.floor(1.2*carw), y  )
        
    car1 = Static_car( position1, vertical=Vertical)
    car2 = Static_car( position2, vertical=Vertical) 
    spritegroup.add( car1 )
    spritegroup.add( car2 )
    
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
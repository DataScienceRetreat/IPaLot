# -*- coding: utf-8 -*-
"""
Created on Mon Mar 5 13:54:37 2018
@author: Orlando Ciricosta

class Car_handler(n):
    returns n moving cars, each having a target parking spot and a
    sprite group of remaing moving cars to collide with. Moving and static
    also have dedicated sprite groups
    
def Populator(spritegroup, car):
    Create a parking spot for car, return its pseudo-wheel target positions,
    and add the static cars created around the spot to the spritegroup

"""

import pygame
import random
from Cars import Car, Static_car
import math


class Car_handler():
    """
    class Car_handler(n):
        returns n moving cars, each having a target parking spot and a
        sprite group of remaing moving cars to collide with. Moving and static
        also have dedicated sprite groups
    """  
    def __init__(self, n):
        
        self.number_of_cars = n
        self.moving_cars = []
        self.collide_with = []
        self.target_positions = []      
        self.static_cars_group = pygame.sprite.Group()
        self.moving_cars_group = pygame.sprite.Group()

        for i in range(n):
            self.moving_cars.append(Car())
            self.collide_with.append(pygame.sprite.Group())
            
            car = self.moving_cars[i]
            self.moving_cars_group.add(car)
            
            # create parking spot and save target positions
            self.target_positions.append(
                    Get_spot(self.static_cars_group, car))
            
        # add each car to the colliding group of the other cars 
        for i in range(n):
            for j in range(n):
                if j != i:
                    self.collide_with[i].add(self.moving_cars[j])

              
                    
#------------------------------------------------------------------------------

def Get_spot(spritegroup, car):
    """
    Create a parking spot for car, return its pseudo-wheel target positions,
    and add the static cars created around the spot to the spritegroup
    """
    #pick random orientation of spot
    Vertical = random.choice([True,False])
    
    screenw = pygame.display.Info().current_w
    screenh = pygame.display.Info().current_h
    
    # sin,cos are used for pseudowheels below
    if Vertical:
        sin = 1
        cos = 0
        carw = car.rect.w
        carh = car.rect.h
    else:
        sin = 0
        cos = 1
        carw = car.rect.h
        carh = car.rect.w
    
    #generate random position of the spot
    x = random.randint( 3*carw, screenw - 3*carw )
    y = random.randint( 3*carh, screenh - 3*carh )
    
    #calculate target pseudo-wheels positions
    frontwheel_x = x + 0.5*car.wheelbase * cos
    frontwheel_y = y + 0.5*car.wheelbase * sin
        
    rearwheel_x = x - 0.5*car.wheelbase * cos
    rearwheel_y = y - 0.5*car.wheelbase * sin
    
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
    
    return ([frontwheel_x, frontwheel_y], [rearwheel_x, rearwheel_y])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
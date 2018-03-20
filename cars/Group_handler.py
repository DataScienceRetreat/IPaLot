# -*- coding: utf-8 -*-
"""
Created on Mon Mar 5 13:54:37 2018
@author: Orlando Ciricosta

class Car_handler(n):
    returns n moving cars, each having a stack of target positions and a
    sprite group of remaing moving cars to collide with. Moving and static
    also have dedicated sprite groups

        self.target_positions[i] = [exit, intermediate target, .. ,
                                    initial target]
    
class Filled_Lot(car_group):
    Creates a filled parking lot at the center of the screen. N_cars will
    be removed from this by the get_spot method, to accomodate the
    incoming moving cars, then the remaining cars will be added to car_group
    for rendering/collisions

"""

import pygame
from .Cars import Car, Static_car
import random
import math
import numpy as np

from .cfg import FROM_BOTTOM, WIDTH, HEIGHT

class Car_handler():
    """
    class Car_handler(n):
        returns n moving cars, each having a stack of target positions and a
        sprite group of remaing moving cars to collide with. Moving and static
        also have dedicated sprite groups
        
        self.target_positions[i] = [exit, intermediate target, .. ,
                                    initial target]
        
    """  
    def __init__(self, n):
        
        self.number_of_cars = n
        self.moving_cars = []
        self.collide_with = []
        self.target_positions = []  # list of stacks of target positions

        self.current_target = []  # these two list the current starting point
        self.current_origin = [] # and target point for each car
        self.car_is_done = [] # will stop updates on the car for the episode

        self.static_cars_group = pygame.sprite.Group()
        self.moving_cars_group = pygame.sprite.Group()
        self.lot = Filled_Lot(self.static_cars_group)
        self.A = self.lot.A
        self.B = self.lot.B
        self.C = self.lot.C
        self.D = self.lot.D        
        
        # now sample n indexes in len(lot.static_cars_list) in order to
        # create the target parking spots (get_spot in the following loop)
        # and add the remaining static cars to the sprite group for
        # collision/rendering
        index = random.sample(range(len(self.lot.static_cars_list)), n)
        for i in range(len(self.lot.static_cars_list)):
            if i not in index:
                self.static_cars_group.add( self.lot.static_cars_list[i] )
                
        # and set the moving cars
        for i in range(self.number_of_cars):
            self.target_positions.append([])
            self.car_is_done.append(False)
            # cars will be at bottom of screen
            car_position = (
                    (i+1) * pygame.display.Info().current_w // (n+1),
                    - pygame.display.Info().current_h + FROM_BOTTOM
                    )
            self.current_origin.append(car_position)
            self.moving_cars.append( Car(pos=car_position) )
            self.collide_with.append(pygame.sprite.Group())
            
            car = self.moving_cars[i]
            self.moving_cars_group.add(car)
            
            # create parking spot and save target positions
            self.lot.get_spot(car, index[i], self.target_positions[i])
            self.current_target.append(self.target_positions[i].pop())
            
            
        # add each car to the colliding group of the other cars 
        for i in range(n):
            for j in range(n):
                if j != i:
                    self.collide_with[i].add(self.moving_cars[j])
                    
    def reset_car(self, i):
        self.moving_cars[i].reset(pos=(
            (i+1) * pygame.display.Info().current_w // (self.number_of_cars+1),
            - pygame.display.Info().current_h + FROM_BOTTOM)
            )
                
    def get_states(self, terminal_flags):
        """ returns a list of states for each car to feed to policy/value NN.
        state[i] is a list [player, target, moving/static]
        ready for feeding into a multi-input NN:
        - player and target have shape (4,), the positions of the pseudowheels
        - moving/static has shape(CAPACITY-1, 4, 1), rect of other cars
        a convolution with few filters will scan the 4-uples for moving/static
        """
        states = []
        for i in range(self.number_of_cars):
            states.append([])
            
            fwp = self.moving_cars[i].get_frontwheel(negative_y = False)
            rwp = self.moving_cars[i].get_rearwheel(negative_y = False)
            player = np.r_[ fwp[0], fwp[1], rwp[0], rwp[1] ]
            
            fwt = self.current_target[i][0]
            rwt = self.current_target[i][1]
            target = np.r_[ fwt[0], fwt[1], rwt[0], rwt[1] ]
            
            # build a list of lists to get the (capacity-1, 4) shaped nparray
            mov_sta = []
            for car in self.collide_with[i]:
                mov_sta.append(list(car.rect))
            for car in self.static_cars_group:
                mov_sta.append(list(car.rect))
            image_mov_sta = np.expand_dims(np.array(mov_sta), axis = 2)
            #this adds a channel dimension to the end of shape (for convnet)
            
            dist1 = self.get_distance(fwp,fwt)
            dist2 = self.get_distance(rwp,rwt)
            if ( dist1 <= 1) and ( dist2 <= 1):
                terminal_flags[i] = True
                if self.target_positions[i]: # if there are new targets left
                    self.current_target[i] = self.target_positions[i].pop()
                else:
                    self.car_is_done[i] = True
                    self.remove_car(i)
            
            states[i].append(player)
            states[i].append(target)            
            states[i].append(image_mov_sta)

            
        return states
    
    def get_distance(self, p1, p2):
        ''' Based on a zone division of the lot, returns a shortest path
        between two points avoiding the self.lot cars. The division goes
        as follows:
                        zone 2
        __________C---------------B_________
                  ----------------
                  ----------------
        zone 3    D---------------A   zone 1
                  |               |
                  |     zone 0    |
            
        if p1 is in zone 0 and p2 too return the distance, if p2 is in zone 1
        return the distance p1-A + A-p2, if p2 is in zone 2 return x-A-B-y
        and so on.
        The general idea is that cars go anticlockwise around the lot
        to reach a destination point
        '''
        distance = 0
        # first determine the zones of p1, p2
        z1 = self.get_zone(p1)
        z2 = self.get_zone(p2)
            
        
        abcd = [self.A, self.B, self.C, self.D]

        if z1 == z2:
            distance = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        else:
            while z1 != z2:
                distance += math.hypot(p1[0]-abcd[z1][0], p1[1]-abcd[z1][1])
                p1 = abcd[z1]
                z1 = (z1+1)%4 # this way if z1 was > z2 to start with, then
                              # the car will just go around anticlockwise
                              # until z1=z2, without index errors
                
            distance += math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                
        return distance
    
    def get_zone(self,p):
        '''get the zone of point p with reference to the get_distance()'''
        if p[1] < self.B[1]:
            return 2    
        else:
            if p[0] > self.A[0]:
                return 1
            elif p[0] < self.D[0]:
                return 3
            else:
                return 0
            
    def remove_car(self, i):
        '''get car[i] out of the game window: this won't create a reset
        because the out-of-boundary condition is checked only for cars
        with car_is_done[i]=False'''
        
        if self.car_is_done[i]: # just to be sure
            self.moving_cars[i].rect.center = 2*WIDTH , 2*HEIGHT
#------------------------------------------------------------------------------

class Filled_Lot():
    ''' Creates a filled parking lot at the center of the screen. N_cars will
        be removed from this by the get_spot method, to accomodate the
        incoming moving cars.
        The design for the matrix of cars is hardcoded: this could be improved
        but YAGNI. The total number of cars on screen will be 30
        
        The lot will have a row of vertical cars at the center of screen which
        will remain untouched, an equal row below, which can contain free
        parking spots, and a row of horizontal cars at the top, which can have
        free spots for parallel parking.
    '''
    
    def __init__(self, car_group):
        ''' static_cars_list will only contain cars in the outer rows,
        and it will be used to add/remove spots, whilst car_group contains
        all of the static cars for collision purposes '''
        self.static_cars_list = []

        self.N = 12     # cars in the central raw, divisible by 2
        n = self.N//2
        
        x = pygame.display.Info().current_w //2
        y = pygame.display.Info().current_h //2 -FROM_BOTTOM
        
        # get dummy car for dimensions
        car = Static_car((x,y))           
        h = car.rect.h
        w = car.rect.w
        
        delta = 1.22*w*0.5 # half position increment for vertical cars
        dx = delta
        
        # during this loop also setup A,B,C,D, the corners of the lot
        # according to the nomenclature used in Car_handler.get_distance()
        for i in range(n):
            #append cars to the sides for the center row
            car1 = Static_car( (x + dx ,y) )
            car2 = Static_car( (x - dx ,y) )            
            car_group.add(car1)
            car_group.add(car2)
            # then for the bottom row
            car1 = Static_car( (x + dx ,y + 1.1*h) )
            car2 = Static_car( (x - dx ,y + 1.1*h) )
            ''' DO NOT add cars in the list to the collider group now
                as they may be removed when assigning free spots '''
            self.static_cars_list.append(car1)
            self.static_cars_list.append(car2)
            if i == n-1:
                self.A = car1.rect.bottomright
                self.D = car2.rect.bottomleft
            # then for the upper row, with horizontal cars
            dx += delta
            if i%2 == 0:
                car1 = Static_car((x + dx ,y - 1.1*(h+w)*0.5), vertical=False)
                car2 = Static_car((x - dx ,y - 1.1*(h+w)*0.5), vertical=False)          
                self.static_cars_list.append(car1)
                self.static_cars_list.append(car2)            
            dx += delta
            if i == n-2:
                self.B = car1.rect.bottomright
                self.C = car2.rect.bottomleft
            
            
    def get_spot(self, car, index, target_list):
        """ Create a parking spot for car and return its pseudo-wheel
        target positions list"""
        
        #first append the exit position to the empty target list
        h = pygame.display.Info().current_h
        exit_pos = (
                (25, h - 25),
                (25, h - 25 - round(car.wheelbase) )
                )
        target_list.append(exit_pos)
        
        spot = self.static_cars_list[index]
        t = spot.get_pseudowheels(car)
        # if the spot is a parallel parking one, set intermediate targets
        # both coming in and getting out (the latter is to simulate a
        # temporary get-out-of-the-way for releasing blocked cars)
        w = car.rect.w
        dx = round(w * 1.22 * 2)
        dy = round(w * 1.22)
        if not spot.is_vertical:
            t_next = ( (t[0][0] - dx, t[0][1] - dy),
                       (t[1][0] - dx, t[1][1] - dy) )
            target_list.append(t_next)
            
        target_list.append(t)
        
        if not spot.is_vertical:       
            target_list.append(t_next)
    
    
    
    
    
    
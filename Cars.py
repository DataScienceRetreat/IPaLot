# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:53:42 2018
@author: Orlando Ciricosta

class Car(pygame.sprite.Sprite, pos=position, wheelbase=wb):
    Provides the controllable object car: the controller can be whoever 
    calls the acceleration\steering methods, either human or AI. The calls
    for human control are in the main loop

class Static_car(pygame.sprite.Sprite):
    provides basic car objects that just seat in
    the given space to provide a collider. For training purposes

def rot_center(image, rect, angle):
    rotate an image while keeping its center

"""

import pygame
from loader import load_image
import math

from cfg import MAXSPEED, MINSPEED, MAXSTEERING, STEERING
from cfg import ACCELERATION, SOFTENING, STEER_SOFTENING


"""
NOTE: I'll keep y negative (standard xy axis, y goes up)
to facilitate the geometry calculations, then use -y when
feeding into rect (imgage xy, y goes down):  rect.y = - self.y
"""

class Car(pygame.sprite.Sprite):
    def __init__(self, pos = None, wheelbase = None):
        pygame.sprite.Sprite.__init__(self)
        self.image = load_image('car_player.png')
        self.rect = self.image.get_rect()
        self.image_orig = self.image
        self.screen = pygame.display.get_surface()
        self.area = self.screen.get_rect()
        
        if pos:
            self.x, self.y = pos
        else:
            self.x = pygame.display.Info().current_w //2
            self.y =  - pygame.display.Info().current_h //2
            
        self.rect.center = self.x, - self.y
        self.dir = 90 # degrees with respect to x axis
        self.speed = 0.0
        self.steer_angle = 0.0
        
        if wheelbase == None:
            self.wheelbase = 0.7 * self.rect.height
        else:
            self.wheelbase = wheelbase

        self.mask = pygame.mask.from_surface(self.image) #for collisions
        
        self.action_list = {
                    0 : [self.accelerate, self.steer_soften],
                    1 : [self.accelerate, self.steerleft],
                    2 : [self.accelerate, self.steerright],
                    3 : [self.deaccelerate, self.steer_soften],
                    4 : [self.deaccelerate, self.steerleft],
                    5 : [self.deaccelerate, self.steerright],
                    6 : [self.soften, self.steer_soften]
                    }

    def act(self, index):
        for func in self.action_list[index]:
            func()

    #bring car to initial position and speed    
    def reset(self, pos=None):
        if pos:
            self.x, self.y = pos
        else:
            self.x = pygame.display.Info().current_w //2
            self.y =  - pygame.display.Info().current_h //2

        self.rect.center = self.x, - self.y
        self.dir = 90
        self.speed = 0.0
        self.steer_angle = 0.0

    #Accelerate the vehicle
    def accelerate(self):
        if self.speed < MAXSPEED:
            self.speed = self.speed + ACCELERATION

    #Deaccelerate
    def deaccelerate(self):
        if self.speed > MINSPEED:
            self.speed = self.speed - ACCELERATION

    #Steer left
    def steerleft(self):
        if self.steer_angle < MAXSTEERING:
            self.steer_angle += STEERING


    #Steer right
    def steerright(self):
        if self.steer_angle > - MAXSTEERING:
            self.steer_angle += -STEERING
        
    #Update the car position
    def update(self):
        # first calculate the virtual wheels' position
        frontwheel_x, frontwheel_y = self.get_frontwheel()
        rearwheel_x, rearwheel_y = self.get_rearwheel()
                
        # then move each virtual wheel in the pointing direction
        rearwheel_x += self.speed * math.cos(math.radians(self.dir))
        rearwheel_y += self.speed * math.sin(math.radians(self.dir))
        
        frontwheel_x += self.speed * math.cos(math.radians(
                self.dir + self.steer_angle))
        frontwheel_y += self.speed * math.sin(math.radians(
                self.dir + self.steer_angle))
        
        # update x and y by averaging the positions of the virtual wheels 
        self.x = 0.5*(rearwheel_x + frontwheel_x)
        self.y = 0.5*(rearwheel_y + frontwheel_y)    
        
        self.rect.center = self.x, - self.y

        # and finally update the car rotation       
        self.dir = math.degrees(
                math.atan2(frontwheel_y - rearwheel_y,
                           frontwheel_x - rearwheel_x)
                )
        
        self.image, self.rect = rot_center(self.image_orig,
                                           self.rect, self.dir )
        self.mask = pygame.mask.from_surface(self.image) #for collisions        
        

    def soften(self):
            if self.speed > 0:
                self.speed = max( 0, self.speed - SOFTENING)
            if self.speed < 0:
                self.speed = min( 0, self.speed + SOFTENING)
                
    def steer_soften(self):
            if self.steer_angle > 0:
                self.steer_angle = max( 0, self.steer_angle - STEER_SOFTENING)
            if self.steer_angle < 0:
                self.steer_angle = min( 0, self.steer_angle + STEER_SOFTENING)
                
    def get_frontwheel(self, negative_y = True):
        frontwheel_x = self.x + 0.5*self.wheelbase * math.cos(
                math.radians(self.dir))
        frontwheel_y = self.y + 0.5*self.wheelbase * math.sin(
                math.radians(self.dir))
        if negative_y:
            return frontwheel_x, frontwheel_y # class xy system
        else:
            return round(frontwheel_x), round(- frontwheel_y) # image xy system
    
    def get_rearwheel(self, negative_y = True):
        rearwheel_x = self.x - 0.5*self.wheelbase * math.cos(
                math.radians(self.dir))
        rearwheel_y = self.y - 0.5*self.wheelbase * math.sin(
                math.radians(self.dir))
        if negative_y:
            return rearwheel_x, rearwheel_y # class xy system       
        else:
            return round(rearwheel_x), round(- rearwheel_y) # image xy system

#------------------------------------------------------------------------------
        
#auxiliary function to rotate a car.
def rot_center(image, rect, angle):
        """rotate an image while keeping its center"""
        rot_image = pygame.transform.rotate(image, angle - 90)
        rot_rect = rot_image.get_rect(center=rect.center)
        return rot_image,rot_rect        
        
        
#------------------------------------------------------------------------------        
        
        
class Static_car(pygame.sprite.Sprite):
    ''' NOTE: in contrast with the moving car, the static car keeps the
        xy system of the screen (no negative y): because of this the y of the
        pseudowheels in get_pseudowheels have different signs
        than the ones used in the corresponding methods for the moving car
        class ''' 
    
    def __init__(self, position, vertical=True):
        pygame.sprite.Sprite.__init__(self)
        self.image = load_image('car6_yellow.png')
        self.rect = self.image.get_rect()
        self.screen = pygame.display.get_surface()
        self.area = self.screen.get_rect()
        self.rect.center = position
        self.is_vertical = vertical
        self.x, self.y = position

        
        if not vertical:
            self.image = pygame.transform.rotate(self.image, 90)
            self.rect = self.image.get_rect(center=self.rect.center)

        self.mask = pygame.mask.from_surface(self.image) #for collisions        
        
    def get_pseudowheels(self, car):
        ''' Returns the pseudowheels positions of a car
        occupying the same positions as self. The reason for passing
        car as an argument is that car may have a different wheelbase '''
        
        wheelbase = car.wheelbase
        if self.is_vertical:
            sin = 1
            cos = 0
        else:
            sin = 0
            cos = 1
            
        frontwheel_x = round(self.x - 0.5*wheelbase * cos)
        frontwheel_y = round(self.y - 0.5*wheelbase * sin)
        
        rearwheel_x = round(self.x + 0.5*wheelbase * cos)
        rearwheel_y = round(self.y + 0.5*wheelbase * sin)
        
        return ((frontwheel_x, frontwheel_y), (rearwheel_x, rearwheel_y)) 
        
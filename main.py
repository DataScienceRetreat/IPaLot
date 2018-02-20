# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:26:37 2018

@author: orlando
"""

import pygame
from pygame.locals import *
import moving_car
from populator import populator

def main():
    
    # Initialise screen
    size = width, height = 700, 700
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Basic parking traing')
    
    # Fill background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    color = (50, 50, 50)
    background.fill(color)
    
    #initialize objects
    car = moving_car.Car()
    player_s  = pygame.sprite.Group()
    player_s.add(car)
    
    static_cars_s = pygame.sprite.Group()
    populator(static_cars_s, car)
    
    
    # Blit everything to the screen
    screen.blit(background, (0, 0))
    pygame.display.flip()

        
    # Event loop
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            
        #Check for key input. (KEYDOWN, trigger often)
        keys = pygame.key.get_pressed()
        if keys[K_LEFT]:
            car.steerleft()
        else:
            car.steer_soften()
            
        if keys[K_RIGHT]:
            car.steerright()
        else:
            car.steer_soften()          
            
        if keys[K_UP]:
            car.accelerate()
        else:
            car.soften()
            
        if keys[K_DOWN]:
            car.deaccelerate()
        else:
            car.soften()
            
        if car.rect.left < 0 or car.rect.right > width:
            car.speed = - car.speed
        if car.rect.top < 0 or car.rect.bottom > height:
            car.speed = - car.speed                  
        
        if pygame.sprite.spritecollide(car, static_cars_s, False, pygame.sprite.collide_mask):
            car.impact()

        car.update()        
            
        #render
       
        screen.blit(background, (0, 0))
        player_s.draw(screen)
        static_cars_s.draw(screen)
        
#        #test subfigure for training
#        reduced = pygame.transform.scale(
#                pygame.display.get_surface(), (100,100))
#        screen.blit(reduced, (0,0))
        
        
        pygame.display.flip()

    
if __name__ == '__main__': main()

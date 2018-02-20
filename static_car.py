# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:53:42 2018
@author: orlando ciricosta

a class providing basic car objects that just seat in
the given space to provide a collider. For training purposes

"""

import pygame
from loader import load_image

class Static_car(pygame.sprite.Sprite):
    def __init__(self, position, vertical=True):
        pygame.sprite.Sprite.__init__(self)
        self.image = load_image('car6_yellow.png')
        self.rect = self.image.get_rect()
        self.screen = pygame.display.get_surface()
        self.area = self.screen.get_rect()
        self.rect.center = position

        
        if not vertical:
            self.image = pygame.transform.rotate(self.image, 90)
            self.rect = self.image.get_rect(center=self.rect.center)

        self.mask = pygame.mask.from_surface(self.image) #for collisions
        
        
        
        
        
        
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:35:34 2018
@author: Orlando Ciricosta

A* objects (Vertex, Priority_queue) for a parking spot search

"""

import heapq as hpq
from itertools import count
from cfg import WIDTH, HEIGHT
from math import hypot
REMOVED = '<removed-task>'     # placeholder for a removed task in a priority-q

#------------------------------------------------------------------------------

class Vertex():
    
    def __init__(self, state, car, parent, parent_action, parent_reward):
        self.id = str(
                [ int(round(state[0][0] * WIDTH)),
                  int(round(state[0][1] * HEIGHT)),
                  int(round(state[0][2] * WIDTH)),
                  int(round(state[0][3] * HEIGHT)),
                  car.speed,
                  car.steer_angle
                ]           
                )
        
        # info to reset the car
        self.car_speed = car.speed
        self.car_steer_angle = car.steer_angle
        self.car_position = car.rect.center
        self.image = car.image
        self.rect = car.rect
        self.mask = car.mask
        self.dir = car.dir
        self.x = car.x
        self.y = car.y
        
        # connections
        if parent:
            self.parent_fw = parent.fw
            self.parent_rw = parent.rw
        else:
            self.parent_fw = None
            self.parent_rw = None
        self.parent_action = parent_action # the action leading to this state
        self.parent_reward = parent_reward # the reward I got ending up here
        
        self.state = state        

        # front and rear wheel for cost to come
        self.fw = [self.state[0][0]*WIDTH, self.state[0][1]*HEIGHT]
        self.rw = [self.state[0][2]*WIDTH, self.state[0][3]*HEIGHT]
        

                                    
                                
    def parent_distance(self):
        return (
            hypot(self.fw[0]-self.parent_fw[0], self.fw[1]-self.parent_fw[1]) +
            hypot(self.rw[0]-self.parent_rw[0], self.rw[1]-self.parent_rw[1])
                )

        
    def update(self, car):
        "update a car's state to the one of the current vertex"
        car.rect.center = self.car_position
        car.speed = self.car_speed
        car.steer_angle = self.car_steer_angle
        car.car_position = car.rect.center
        car.image = self.image
        car.rect = self.rect
        car.mask = self.mask
        car.dir = self.dir
        car.x = self.x
        car.y = self.y
        


#------------------------------------------------------------------------------

class Priority_queue():
    
    def __init__(self):
        self.list = []              # list of entries arranged in a heap
        self.entry_finder = {}    # mapping of tasks to entries
        self.counter = count()    # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        hpq.heappush(self.list, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.list:
            priority, count, task = hpq.heappop(self.list)
            if task is not REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')
        
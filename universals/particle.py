import numpy as np

class Particle:
    
    def __init__(self, coor, vel, rad, m):
        self.coordinate = coor
        self.velocity = vel
        self.radius = rad
        self.mass = m
        self.color = np.random.rand(3,)
    
    def motion_update(self, force, delta_t):
        self.coordinate += self.velocity * delta_t
        self.velocity += (force/self.mass) * delta_t
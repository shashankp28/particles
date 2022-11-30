import numpy as np

def is_close(a, b, eps):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b)<=eps

def get_boundary(circles, fuzz):
    d_max = -float('inf')
    max_radius = d_max
    particle_coors = []
    for (x, y, radius) in circles: 
        d_max = max([d_max, abs(x)+radius, abs(y)+radius])
        max_radius = max(max_radius, radius)
        particle_coors.append((x*fuzz, y*fuzz, radius))
    return d_max, max_radius, particle_coors

def get_force(p1, p2):
    return p1.mass * p2.mass * (p2.coordinate - p1.coordinate) / ((np.linalg.norm(p2.coordinate - p1.coordinate))**3)

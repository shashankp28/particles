import math
import numpy as np
import packcircles as pc
from services.utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from universals.particle import Particle

class Environment:
    
    def __init__(self, dt, mv, m_a, n, G, sc, sp):
        self.D_max = None
        self.particles = []
        self.delta_t = dt
        self.max_vel = mv
        self.M_by_area = m_a
        self.N = n
        self.G = G
        self.collision_times = []
        self.scale = sc
        self.spread = sp
        
    def generate_particles(self):
        print("Generating Particles...")
        radii = [np.random.uniform(0.2, 0.4) for _ in range(self.N)]
        circles = pc.pack(radii)
        self.D_max, max_radius, particle_coors = get_boundary(circles, self.spread)
        self.D_max = (self.D_max + max_radius)*2 * self.scale
        for (x, y, radius) in particle_coors:
            mass = radius*self.M_by_area*math.pi
            temp_coor = np.array([x, y]) + np.array([self.D_max/2, self.D_max/2])
            temp_vel = np.random.uniform(-1, 1, (2,))*self.max_vel
            particle = Particle(temp_coor, temp_vel, radius, mass)
            self.particles.append(particle)

    def show_environment(self, i):
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_xlim([0, self.D_max])
        ax.set_ylim([0, self.D_max])
        energy = 0
        momentum = np.zeros(2,)
        for p in self.particles:
            momentum += p.mass * p.velocity
            energy += 0.5 * p.mass * np.dot(p.velocity, p.velocity)
            circle = plt.Circle(p.coordinate, p.radius, color=p.color)
            ax.add_patch(circle)
        ax.set_title('Kinetic Energy: {} & Momentum: {}'.format(np.round(energy, 3), np.round(momentum, 3)))
        fig.savefig("./images/{}.jpg".format(i))
        plt.close(fig)
    
    def fix_border(self, particle):
        collided = False
        x, y = particle.coordinate
        if is_close((x), (self.D_max), particle.radius) or is_close(x, 0, particle.radius): 
            particle.velocity = np.multiply(particle.velocity, np.array([-1, 1]))
            particle.just_border = True
            collided = True
        if is_close((y), (self.D_max), particle.radius) or is_close(y, 0, particle.radius): 
            particle.velocity = np.multiply(particle.velocity, np.array([1, -1]))
            particle.just_border =  True
            collided = True
        return particle, collided
    
    def check_collisions(self):
        collided = False
        for i in range(len(self.particles)-1):
            for j in range(i+1, len(self.particles)):
                p1 = self.particles[i]
                p2 = self.particles[j]
                x1 = p1.coordinate
                x2 = p2.coordinate
                dist = p1.radius + p2.radius
                if is_close(x1, x2, dist):
                    collided = True
                    m1 = p1.mass
                    m2 = p2.mass
                    v1 = p1.velocity
                    v2 = p2.velocity
                    dist = np.linalg.norm(x1-x2)
                    self.particles[i].velocity = v1 - (2*m2/(m1+m2)) * np.dot(v1-v2, x1-x2) * (1/(dist**2)) * (x1 - x2)
                    self.particles[j].velocity = v2 - (2*m1/(m1+m2)) * np.dot(v2-v1, x2-x1) * (1/(dist**2)) * (x2 - x1)
        return collided            
    
    def update_environment(self, fd):
        
        force_matrix = np.zeros((self.N, self.N, 2))
        
        collision_border = False
        
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                force_matrix[i][j] = self.G * get_force(self.particles[i], self.particles[j])
        
        individual_forces = np.zeros((self.N, 2))
        
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                individual_forces[i] += force_matrix[i][j]
                individual_forces[j] -= force_matrix[i][j]
        
        for i, p in enumerate(self.particles): 
            p.motion_update(individual_forces[i], self.delta_t)
            p, col_border = self.fix_border(p)
            collision_border = collision_border or col_border
        
        col_other = self.check_collisions()
        if collision_border or col_other: self.collision_times.append(fd)
        return
            
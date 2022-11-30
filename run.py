import os
import cv2
import math
import shutil
from tqdm import tqdm
from universals.environment import Environment

img_path = "./images/"


shutil.rmtree(img_path)
os.mkdir(img_path)

fps = 60
duration = 60

env = Environment(dt=1/fps, mv=2, m_a=0.5, n=20, G=1, sc=5, sp=6)
env.generate_particles()
i = 0
print("Running Simulation...")


for i in tqdm(range(fps*duration)):
    env.update_environment(i)
    env.show_environment(i)
    i+=1

img_array = []
files = os.listdir(img_path)
files = sorted(files, key=lambda x: int(x.split('.')[0]))

print("Reading Images...")

for filename in tqdm(files):
    img = cv2.imread(img_path+filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

print("Making The video...") 
for i in tqdm(range(len(img_array))):
    out.write(img_array[i])
out.release()
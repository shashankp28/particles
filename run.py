import os
import cv2
import shutil
from tqdm import tqdm
from universals.environment import Environment

img_path = "./images/"

try:
    shutil.rmtree(img_path)
except:
    pass
os.mkdir(img_path)

fps = 60
duration = 300

env = Environment(dt=1/fps, mv=1, mrr=3, n=10, G=0.5)
env.generate_particles()
i = 0
print("Running Simulation...")


for i in tqdm(range(fps*duration)):
    env.update_environment()
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
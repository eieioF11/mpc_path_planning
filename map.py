import cv2
import numpy as np

grid=np.array([
  [0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0],
  [0,0,1,1,0,0,0,0,0,0,0],
  [0,0,1,1,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,1,1,0,0,0],
  [0,0,0,0,0,0,1,1,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0]
])
map_resolution = 1
origin = [0,0]
#変換
def conversion_grid_pos(map_pos):
    return [int(map_pos[0]/map_resolution+origin[0]),int(map_pos[1]/map_resolution+origin[1])]
def conversion_map_pos(grid_pos):
    return [(grid_pos[0]-origin[0])*map_resolution,(grid_pos[1]-origin[1])*map_resolution]

indices = np.dstack(np.where(grid == 1))
#print(indices)
for p in indices:
  pos=conversion_map_pos(p)
  print(pos)

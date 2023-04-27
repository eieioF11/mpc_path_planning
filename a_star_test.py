import pathplanner.a_star as a_star
import sys
import numpy as np
import math
import matplotlib.pyplot as plt


def main():
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
  start=np.array([0,0])
  goal=np.array([9,9])
  grid_path_planner = a_star.PathPlanner(grid,False)
  init,path=grid_path_planner.a_star(start,goal)
  if init!=-1:
    path=np.vstack((init,path))#初期位置をpathに追加
    #結果表示
    print(path)
    plt.imshow(grid)
    plt.plot(path[:,1],path[:,0])
    plt.show()
  else :
    print('error')

if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit()
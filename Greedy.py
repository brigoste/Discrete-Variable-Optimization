import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

# ------------------------------- HEADER ---------------------------------------------------------
# Author: Brigham Ostergaard
# Title: Greedy Algorithm
# Date: 3/26/2025
# Descritpion: 
#   A space is defined and points are randomly set to populate the space. Starting from the point (0,0),
#       A path is created using a greedy algorithm to connect all the dots and then go back to the start.

def next_point(prev_point,points):
    # find the next closest point to our current point
    smallest_dist = np.inf
    for i in range(np.shape(points)[0]):
        dist = np.sqrt((points[i,0] - prev_point[0])**2 + (points[i,1] - prev_point[1])**2) 
        if(dist < smallest_dist):
            smallest_dist = dist
            next_point = i
    
    
    return next_point,smallest_dist        # index of the next point

def Greedy_path(points):
    start_point = np.array([0,0])
    sort_points = np.array([])
    curr_point = start_point    # point 1 (current point or start)
    track_dist = 0                  # total distance traveled

    for i in range(np.shape(points)[0]):
        # find index of next point in points
        next_ind,dist = next_point(curr_point,points)
        # add the next point to our storted array
        curr_point = points[next_ind,:]     #store current point
        if(i == 0):
            sort_points = curr_point
        else:
            sort_points = np.vstack([sort_points,curr_point])
        # remove curr_point from points (at the next index)
        points = np.delete(points,next_ind,axis=0)
        # add to our cumulative distance
        track_dist += dist

    # add distance from last point to first point
    dist = np.sqrt((start_point[0] - sort_points[-1,0])**2 + (start_point[1]-sort_points[-1,1])**2)
    track_dist += dist
    # add the beginign and end points the path
    sort_points = np.vstack([start_point,sort_points,start_point]) # go back to start

    return sort_points,track_dist         # indecies, in order of the array.

def plot_path(points):
    plt.figure()
    plt.scatter(points[1:-1,0],points[1:-1,1])  # plot intermediary points
    plt.plot(points[:,0],points[:,1],'r')       # connect points with path
    plt.scatter(points[0,0],points[0,1], 120, color ='g', marker="*", label="start\end") # show first/last point
    plt.legend(loc='best')
    plt.show()

def main():
    lower_bound = -10
    upper_bound = 10
    n = 150
    points = np.random.uniform(lower_bound, upper_bound, [n-1,2])       # n-1 because I start and end at [0,0] no matter what
    show_plots = True

    sorted_points,dist = Greedy_path(points)

    print("Original Points:\n",points)
    print("Sorted Points:\n",sorted_points)
    print(f"Total distance = {dist}")

    if(show_plots):
        plot_path(sorted_points)



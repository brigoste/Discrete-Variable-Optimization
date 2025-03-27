import numpy as np
import scipy as sci
from Greedy import Greedy_path
import matplotlib.pyplot as plt

# ------------------------------- HEADER ---------------------------------------------------------
# Author: Brigham Ostergaard
# Title: Simulated Annealing
# Date: 3/26/2025
# Descritpion: 
#   A space is defined and points are randomly set to populate the space. Starting from the point (0,0),
#       A path is created using the greedy algorithm to connect all the dots and then go back to the start.
#       The path is then modified using simmulated annealing to (hoepfully) get an even better solution. The paths
#       for greedy and the simulated annealing are both shown and the final path length for each is compared.


def measure_path(points):
    # return the distance along the path for all points
    n = np.shape(points)[0]
    dist = 0
    for i in range(n-1):
        dist += np.sqrt((points[i,0] - points[i+1,0])**2 + (points[i,1] - points[i+1,1])**2)
    
    return dist

def neighbor_swap(points):
    n = np.shape(points)[0]
    group_me = False

    # Method 1: select a random point and switch it with one of the 4 closest other points
    if(group_me == False):
        # select a random index
        rand_index = (int)(np.random.uniform(2,n-2))

        # find 4 closest points to our index.
        shortest_distance = np.inf
        shortest_distance2 = np.inf
        shortest_distance3 = np.inf
        shortest_distance4 = np.inf
        for i in range(n-1):
            if(i != rand_index):
                dist = np.linalg.norm(points[i+1,:] - points[rand_index,:])
                if(dist < shortest_distance):
                    point_close_i = i
                    shortest_distance = dist
                elif(dist < shortest_distance2):
                    point_close_i2 = i
                    shortest_distance2 = dist
                elif(dist < shortest_distance3):
                    point_close_i3 = i
                    shortest_distance3 = dist
                elif(dist < shortest_distance4):
                    point_close_i4 = i
                    shortest_distance4 = dist

        # choose randomly which of the points will swap with original random index
        rand_dir = np.random.rand()
        if(rand_dir >= 0.75):
            points[rand_index,:], points[point_close_i,:] = points[point_close_i,:].copy(), points[rand_index,:].copy()
        elif(rand_dir >= 0.5):
            points[rand_index,:], points[point_close_i2,:] = points[point_close_i2,:].copy(), points[rand_index,:].copy()
        elif(rand_dir > 0.25):
            points[rand_index,:], points[point_close_i3,:] = points[point_close_i3,:].copy(), points[rand_index,:].copy()
        else:
            points[rand_index,:], points[point_close_i4,:] = points[point_close_i4,:].copy(), points[rand_index,:].copy()
    
    # Method 2: move a neighborhood of random size to a differnt part of the order
    else:
        # This kind of works, but heres another way
        rand_index = (int)(np.random.uniform(2,n-2))
        rand_length = (int)(np.random.uniform(0.3*n))
        if(rand_index + rand_length >= n):
            rand_length = n - rand_index - 1
        
        org_size = np.shape(points)[0]
        # get a small section of points
        neighborhood = points[rand_index:rand_index+rand_length,:]

        # find a new location to slot those points    
        rand_insert = (int)(np.random.uniform(2,n-2-rand_length))
        # delete the points from their existing location in the array
        # Delete the points from their existing location in the array
        points = np.delete(points, np.arange(rand_index, rand_index + rand_length), axis=0)
        # insert the neighborhood at the new location.
        points = np.insert(points,rand_insert,neighborhood,axis=0)

        new_size = np.shape(points)[0]

    return points       # return points with the switches

def plot_steps(points):
    plt.ion()
    plt.scatter(points[:,0],points[:,1])
    plt.plot(np.hstack([points[:,0],points[0,0]]),np.hstack([points[:,1],points[0,1]]))
    plt.pause(0.005)
    plt.cla()


lower_bound = -10
upper_bound = 10
n = 30
plot_each_step = True
points = np.random.uniform(lower_bound, upper_bound, [n-1,2])       # n-1 because I start and end at [0,0] no matter wha

# Start with the greedy algorithm to get an initial dataset
points_greedy,dist = Greedy_path(points.copy())

print(points_greedy)

max_iter = 25000
T0 = 1000
T = T0
Beta = 4 # on the range (1-4), higher number means we keep the temperature lower longer.

# original path
hold_points_greedy = points_greedy.copy()
f_store = np.array([measure_path(hold_points_greedy)])

points = points_greedy.copy()
global count_swaps

count_swaps = 0

# ----------------------Start the annealing loop------------------------------
def Simulated_annealing(points,f_store,T0,plot_each_step=True):
    global count_swaps
    T = T0
    for i in range(max_iter):
        if(i%10 == 0):
            print(f"Iteration {i}")
        
        # switch order between 2 random neighboring points
        points_new = neighbor_swap(points.copy())       # looks at a random set of points and puts them in a new location order
        # if the new points have a shorter distance, than there is a chance these become the new dataset.

        if(np.linalg.norm(points - points_new) > 0):
            count_swaps +=1

        if(measure_path(points_new) <= measure_path(points)):
            points = points_new.copy()
        else:
            r = np.random.rand()
            num = -(measure_path(points_new)-measure_path(points))
            P = np.exp(num / T)

            if(P >= r):
                points = points_new.copy()

        f_store = np.hstack([f_store,measure_path(points)])
        T = T0*(1-(i/max_iter))**Beta

        if(plot_each_step and i%50 == 0):
            plot_steps(points)

    return points,f_store

points,f_store = Simulated_annealing(points,f_store,T0,plot_each_step)

if(f_store[-1] > f_store[0]):
    points,f_store = Simulated_annealing(points,f_store,T0,plot_each_step)

plt.ioff()
plt.close()
print(f"Iteration {max_iter}")
dist_f = measure_path(points)

sim_plot_points = np.vstack([points,points[0,:]])

print(f"\nGreedy distance = {f_store[0]}\nSimulated Annealing = {f_store[-1]}")
print(f"Swaps = {count_swaps}")
print(f"Final T = {T}")
plt.figure()
plt.scatter(points[:,0],points[:,1])
plt.plot(sim_plot_points[:,0],sim_plot_points[:,1],'r')
plt.scatter(points[0,0],points[0,1], 120, color ='g', marker="*", label="start\end")            # show first point
plt.legend(loc='best')
plt.title('Simulated Annealing')

plt.figure()
plt.scatter(points_greedy[1:-1,0],points_greedy[1:-1,1])
plt.plot(points_greedy[:,0],points_greedy[:,1],'r')
plt.scatter(points_greedy[0,0],points_greedy[0,1], 120, color ='g', marker="*", label="start\end")            # show first point
plt.legend(loc='best')
plt.title("Greedy Algorithm")

plt.figure()
plt.semilogy(np.linspace(0,1,(2*max_iter)+1),f_store)
plt.xlabel('Iteration')
plt.ylabel('f')

plt.show()
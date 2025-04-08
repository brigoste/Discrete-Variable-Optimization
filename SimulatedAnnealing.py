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

# ----------------------Start the annealing loop------------------------------
def Simulated_annealing(points,f_store,T0,Beta,show_plots=True,max_iter=25000):
    global count_swaps
    T = T0
    for i in range(max_iter):
        # if(i%1000 == 0):
        #     print(f"Iteration {i}")
        
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

        if(show_plots and i%50 == 0):
            plot_steps(points)

    return points,f_store,max_iter

def plot_path(points,sim_plot_points,points_greedy,max_iter,f_store):
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
    plt.semilogy(np.linspace(0,1,max_iter+1),f_store)
    plt.xlabel('Iteration')
    plt.ylabel('f')

    plt.show()

def plot_path2(points,sim_plot_points,iteration):
    figure_title = f"ME 575  HW8\\Figures\\Simulated_Annealing_Iteration_{iteration}.jpg"
    plt.figure()
    plt.scatter(points[:,0],points[:,1])
    plt.plot(sim_plot_points[:,0],sim_plot_points[:,1],'r')
    plt.scatter(points[0,0],points[0,1], 120, color ='g', marker="*", label="start\end")            # show first point
    plt.legend(loc='best')
    plt.title(f'n = {iteration}')
    plt.savefig(figure_title, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_path3(points,sim_plot_points,iteration, Beta,T0):
    figure_title = f"ME 575  HW8\\Figures2\\Simulated_Annealing_Iteration_{iteration}.jpg"
    plt.figure()
    plt.scatter(points[:,0],points[:,1])
    plt.plot(sim_plot_points[:,0],sim_plot_points[:,1],'r')
    plt.scatter(points[0,0],points[0,1], 120, color ='g', marker="*", label="start\end")            # show first point
    plt.legend(loc='best')
    plt.title(f'n = {iteration}, Beta = {Beta}, x0 = [{np.round(points[0,0],2)},{np.round(points[0,1],2)}], T0 = {T0}')
    plt.savefig(figure_title, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_path4(points,sim_plot_points,iteration, Beta,i):
    figure_title = f"ME 575  HW8\\Figures3\\Simulated_Annealing_Iteration_{iteration}_i_{i}_Beta_{Beta}.jpg"
    plt.figure()
    plt.scatter(points[:,0],points[:,1])
    plt.plot(sim_plot_points[:,0],sim_plot_points[:,1],'r')
    plt.scatter(points[0,0],points[0,1], 120, color ='g', marker="*", label="start\end")            # show first point
    plt.legend(loc='best')
    plt.title(f'n = {iteration}, Beta = {Beta}, x0 = [{np.round(points[0,0],2)},{np.round(points[0,1],2)}], shift = {i}')
    plt.savefig(figure_title, dpi=300, bbox_inches='tight')
    # plt.show()

def main():
    show_plots = True
    lower_bound = -10
    upper_bound = 10
    n = 30
    points = np.random.uniform(lower_bound, upper_bound, [n-1,2])       # n-1 because I start and end at [0,0] no matter wha
    points_orig = points.copy()
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
    total_iterations = 0
    total_iterations2 = 0

    points,f_store,total_iterations = Simulated_annealing(points,f_store,T0,Beta,show_plots,max_iter)

    # what if this yields a path worse than the greedy algorithm? 
    # Restart seeding the first iteration with the last iteration of the previous run of simulated_annealing.    
    if(f_store[-1] > f_store[0]):   
        points,f_store,total_iterations2 = Simulated_annealing(points,f_store,T0,Beta,show_plots,max_iter)

    iterations = total_iterations+total_iterations2

    plt.ioff()
    plt.close()
    print(f"Iteration {iterations}")
    dist_f = measure_path(points)

    sim_plot_points = np.vstack([points,points[0,:]])

    print(f"\nGreedy distance = {f_store[0]}\nSimulated Annealing = {f_store[-1]}")
    print(f"Swaps = {count_swaps}")
    print(f"Final T = {T}")

    if(show_plots):
        plot_path(points,sim_plot_points,points_greedy,iterations,f_store)

def repeat_annealing():
    show_plots = False
    lower_bound = -10
    upper_bound = 10
    n = 49
    points = np.random.uniform(lower_bound, upper_bound, [n-1,2])       # n-1 because I start and end at [0,0] no matter wha
    points_orig = points.copy()
    # Start with the greedy algorithm to get an initial dataset
    points_greedy,dist = Greedy_path(points.copy())

    max_iter = 10000
    T0 = 1000
    T = T0
    Beta = 4 # on the range (1-4), higher number means we keep the temperature lower longer.
    f_store = np.array([])
    f_best = np.array([])

    for i in range(100):
        print(f"Iteration {i}")
        global count_swaps
        count_swaps = 0
        points,f_store,total_iterations = Simulated_annealing(points_orig,f_store,T0,Beta,show_plots,max_iter)
        f_best = np.hstack([f_best,f_store[-1]])
        if(i == 29):
            sim_plot_points = np.vstack([points,points[0,:]])
            plot_path2(points,sim_plot_points,i,Beta,T0)

    print(f"Best distances = {f_best}")
    plt.figure()
    plt.plot(np.linspace(1,100,100),f_best)
    plt.xlabel('Iteration')
    plt.ylabel('f')
    plt.title('Best Distance')
    plt.savefig("ME 575  HW8\\Figures\\Simulated_Annealing_Best_Distances.jpg", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Best distance = {np.min(f_best)}")

def compare_annealing():
    show_plots = False
    lower_bound = -10
    upper_bound = 10
    n = 49
    points = np.random.uniform(lower_bound, upper_bound, [n-1,2])       # n-1 because I start and end at [0,0] no matter wha
    points_orig = points.copy()
    # Start with the greedy algorithm to get an initial dataset
    points_greedy,dist = Greedy_path(points.copy())

    T0 = 1000
    T = T0
    show_plots = False
    f_store = np.array([])

    # we want to loop this for different values of Beta and max_iter to see how it changes the results.
    # I want to also change the starting point
    for i in range(3):
        print(f'Running with {i} shifts')
        global count_swaps
        count_swaps = 0
        if(i > 0):
            points_start = np.roll(points_greedy,-i,axis=0)    # shift points one spot
            points_start = np.delete(points_start,-i,axis=0)
            points_start = np.vstack([points_start,points_start[0,:]])
        else:
            points_start = points_greedy.copy()
        for max_iter in [10000,25000,50000]:
            points = points_start.copy()
            Beta = 4
            points_fast,f_store,total_iterations = Simulated_annealing(points_start,f_store,T0,Beta,show_plots,max_iter)
            plot_path4(points_fast,points_fast,max_iter,Beta,i)
            Beta = 1
            points_slow,f_store2,total_iterations2 = Simulated_annealing(points_start,f_store,T0,Beta,show_plots,max_iter)
            plot_path4(points_slow,points_slow,max_iter,Beta,i)

    max_iter = 10000
 
    Beta = 4 # on the range (1-4), higher number means we keep the temperature lower longer.
    f_store = np.array([])
    f_best = np.array([])

    n_starts = 30

    for i in range(n_starts):
        print(f"Iteration {i}")
        count_swaps = 0
        points,f_store,total_iterations = Simulated_annealing(points_orig,f_store,T0,Beta,show_plots,max_iter)
        f_best = np.hstack([f_best,f_store[-1]])
        if(i == 29):
            sim_plot_points = np.vstack([points,points[0,:]])
            plot_path3(points,sim_plot_points,i,Beta,T0)

    print(f"Best distances = {f_best}")
    plt.figure()
    plt.plot(np.linspace(1,n_starts,n_starts),f_best)
    plt.xlabel('Iteration')
    plt.ylabel('f')
    plt.title('Best Distance')
    plt.savefig("ME 575  HW8\\Figures2\\Simulated_Annealing_Best_Distances.jpg", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Best distance = {np.min(f_best)}")


global count_swaps
# main()
# repeat_annealing()
compare_annealing()

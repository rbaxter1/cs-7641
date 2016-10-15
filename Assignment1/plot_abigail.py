import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import uuid
import numpy as np
import pandas as pd


def plot(df, series_name):
    plt.clf()
    
    x = df.ix[:,0]
    train = df.ix[:,1]
    test = df.ix[:,2]
    #time = df.ix[:,2]
    
    plt.plot(x, train,
             color='blue', marker='o',
             markersize=0,
             label='training error')
        
    plt.plot(x, test,
             color='green', marker='s',
             markersize=0, linestyle='--',
             label='testing error')
    
    
    plt.grid()
    plt.title("Train vs. Test Error: %s (Titanic)" % (series_name))
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend(loc='best')
    fn = './output/' + str(uuid.uuid4()) + '_' + series_name + '_traintest.png'
    plt.savefig(fn)


def plot2(df, algo_name, series_name):
    plt.clf()
    
    x = df.ix[:,0]
    opt = df.ix[:,1]
    time = df.ix[:,3]
    
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    
    ax1.plot(x, opt, 'b-')
#             color='blue', marker='o',
#             markersize=0,
#             label='Optima')
    ax2.plot(x, time, 'g-')
#             color='green', marker='s',
#             markersize=0, linestyle='--',
#             label='time')

    #ax1.plot(x, y1, 'g-')
    #ax2.plot(x, y2, 'b-')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Optimal Solution', color='g')
    ax2.set_ylabel('Time (in Seconds)', color='b')

    plt.grid()
    plt.title("%s (%s)" % (algo_name, series_name))
    #plt.xlabel('Iteration')
    #plt.ylabel('Optimal Solution')
    plt.legend(loc='best')
    fn = './output/' + str(uuid.uuid4()) + '_' + series_name + '_' + algo_name + '_iterations.png'
    plt.savefig(fn)

if __name__ == "__main__":
    '''
    df1 = pd.read_csv('./data/RHC.txt', sep=',')
    df2 = pd.read_csv('./data/SA.txt', sep=',')
    df3 = pd.read_csv('./data/GA.txt', sep=',')
    df4 = pd.read_csv('./data/BackProp.txt', sep=',')
    plot(df1, 'Random Hill Climbing')
    plot(df2, 'Simulated Annealing')
    plot(df3.ix[0:800,:], 'Genetic Algorithm')
    plot(df4, 'Back Propagation')
    '''
    
    '''
    df5 = pd.read_csv('./data/Knapsack_RHC.txt', sep=',')
    df6 = pd.read_csv('./data/Knapsack_SA.txt', sep=',')
    df7 = pd.read_csv('./data/Knapsack_GA.txt', sep=',')
    df8 = pd.read_csv('./data/Knapsack_MIMIC.txt', sep=',')
    plot2(df5.ix[0:500,:], 'Random Hill Climbing', 'Knapsack Problem')
    plot2(df6.ix[0:500,:], 'Simulated Annealing', 'Knapsack Problem')
    plot2(df7.ix[0:25000,:], 'Genetic Algorithm', 'Knapsack Problem')
    plot2(df8.ix[0:500,:], 'MIMIC', 'Knapsack Problem')
    '''
    
    df5 = pd.read_csv('./data/TSP_RHC.txt', sep=',')
    df6 = pd.read_csv('./data/TSP_SA.txt', sep=',')
    df7 = pd.read_csv('./data/TSP_GA.txt', sep=',')
    df8 = pd.read_csv('./data/TSP_MIMIC.txt', sep=',')
    
    
    plot2(df5.ix[0:20000,:], 'Random Hill Climbing', 'Travelling Salesman Problem')
    plot2(df6.ix[0:20000,:], 'Simulated Annealing', 'Travelling Salesman Problem')
    plot2(df7.ix[0:5000,:], 'Genetic Algorithm', 'Travelling Salesman Problem')
    plot2(df8.ix[0:8000,:], 'MIMIC', 'Travelling Salesman Problem')
    
        
    
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

class plot_helper():
    def __init__(self):
        pass
    
    def get_arrow(self, grid_policy, x, y):
        if grid_policy[y, x] == 0:
            return '^'
        if grid_policy[y, x] == 1:
            return 'v'
        if grid_policy[y, x] == 2:
            return '<'
        if grid_policy[y, x] == 3:
            return '>'
    
    def plot_heatmap(self, grid, grid_labels, grid_policy, title, filename):
        plt.clf()
        plt.cla()
        '''
        pylab.plot(x,y)
        F = pylab.gcf()
        DPI = F.get_dpi()
        print("DPI:", DPI)
        DefaultSize = F.get_size_inches()
        print("Default size in Inches", DefaultSize)
        print("Which should result in a %i x %i Image"%(DPI*DefaultSize[0], DPI*DefaultSize[1]))
        '''
        cell_scaler = 0.5
        figsize=(max(1, int(grid.shape[0]*cell_scaler*(8/6))), max(1, int(grid.shape[1]*cell_scaler)))
        plt.figure(figsize=figsize, dpi=96)
        
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

        frame1.axes.xaxis.set_visible(False)
        frame1.axes.yaxis.set_visible(False)
        
        heatmap = plt.pcolor(grid)
        
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                '''
                self.space_end = 2
                self.space_void = 3
                self.space_quicksand = 4
                '''
                
                if grid_labels[y, x] == 1:
                    # start
                    str = '%.2f' % grid[y, x] + '\nS ' + self.get_arrow(grid_policy, x, y)
                    
                    plt.text(x + 0.5, y + 0.5, str,
                             horizontalalignment='center',
                             verticalalignment='center')
                    
                elif grid_labels[y, x] == 4:
                    # quicksand
                    str = '%.2f' % grid[y, x] + '\nQ ' + self.get_arrow(grid_policy, x, y)
                    
                    plt.text(x + 0.5, y + 0.5, str,
                             horizontalalignment='center',
                             verticalalignment='center')
                    
                elif grid_labels[y, x] == 3:
                    # obstacle
                    str = '%.2f' % grid[y, x] + '\nX ' + self.get_arrow(grid_policy, x, y)
                    
                    plt.text(x + 0.5, y + 0.5, str,
                             horizontalalignment='center',
                             verticalalignment='center')
                    
                elif grid_labels[y, x] == 2:
                    # obstacle
                    str = '%.2f' % grid[y, x] + '\nG ' + self.get_arrow(grid_policy, x, y)

                    plt.text(x + 0.5, y + 0.5, str,
                             horizontalalignment='center',
                             verticalalignment='center')
        
                else:
                    str = '%.2f' % grid[y, x] + '\n  ' + self.get_arrow(grid_policy, x, y)
                    
                    plt.text(x + 0.5, y + 0.5, str,
                             horizontalalignment='center',
                             verticalalignment='center')
        
        plt.colorbar(heatmap)
        plt.title(title)
        

        plt.savefig(filename)


    def plot_grid(self, grid, xlab, ylab, title, filename):
        plt.clf()
        plt.cla()
        '''
        cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.5, 1.0, 0.7),
                     (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0),
                       (0.5, 1.0, 0.0),
                       (1.0, 1.0, 1.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (0.5, 1.0, 0.0),
                     (1.0, 0.5, 1.0))}
        cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
        '''
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['green','white','red'],
                                           256)
        
        img = plt.imshow(grid,interpolation='nearest',
            cmap = cmap,
            origin='lower')
        
        plt.colorbar(img,cmap=cmap)

        #plt.grid()
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        
        plt.savefig(filename)
        

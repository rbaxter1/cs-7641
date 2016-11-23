import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib._png import read_png
import matplotlib.colors as colors
import matplotlib.ticker as ticker

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
    
    def __calculate_figsize(self, rows, cols, cell_scaler=0.5):
        figsize = (max(1, int(rows*cell_scaler*(8/6))), max(1, int(cols*cell_scaler)))
        return figsize
    
    def __my_colormap(self):
        '''
        cdict = {
            'red'  :  ( (0.0, 0.25, 0.25), (0.02, 0.59, 0.59), (1.00, 1.00, 1.00)),
            'green':  ( (0.0, 0.00, 0.00), (0.02, 0.45, 0.45), (1.00, 0.97, 0.97)),
            'blue' :  ( (0.0, 1.00, 1.00), (0.02, 0.75, 0.75), (1.00, 0.45, 0.45)) }
        
        return mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
        '''
        #return mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
        #                                                    ['green','white','red'],
        #                                                    256)
        #return cm = cm.viridis()
        
    def plot_series(self, x, y, y_std, y_lab, colors, markers, title, xlab, ylab, filename):
        
        plt.clf()
        plt.cla()
        fig, ax = plt.subplots()
        
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        for i in range(len(y)):
            plt.plot(x, y[i],
                     color=colors[i], marker=markers[i],
                     markersize=5,
                     label=y_lab[i])
            
            if None != y_std[i]:
                plt.fill_between(x,
                                 y[i] + y_std[i],
                                 y[i] - y_std[i],
                                 alpha=0.15, color=colors[i])
        
        plt.grid()
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend(loc='best')
        
        plt.savefig(filename)
        
    def plot_layout(self, grid, title, filename):
        plt.clf()
        plt.cla()
        
        nrows, ncols = grid.shape
        
        # resize
        figsize = self.__calculate_figsize(nrows, ncols)
        plt.figure(figsize=figsize, dpi=96)
        
        cmap = colors.ListedColormap(['white', 'green', 'blue', 'black', 'red', 'gold'])
        
        row_labels = range(nrows)
        col_labels = range(nrows)
        plt.matshow(grid, extent=[0, nrows, 0, ncols], cmap=cmap)
                    
        plt.xticks(range(ncols), col_labels)
        plt.yticks(range(nrows), row_labels)
        
        # hide axes
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

        plt.grid()
        plt.title(title)
        
        plt.savefig(filename)

    def plot_heatmap_simple(self, grid, title, filename):
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
        
        nrows, ncols = grid.shape
        
        # resize
        figsize = self.__calculate_figsize(nrows, ncols)
        plt.figure(figsize=figsize, dpi=96)
        
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

        frame1.axes.xaxis.set_visible(False)
        frame1.axes.yaxis.set_visible(False)
        '''
        cdict = {
            'red'  :  ( (0.0, 0.25, 0.25), (0.02, 0.59, 0.59), (1.00, 1.00, 1.00)),
            'green':  ( (0.0, 0.00, 0.00), (0.02, 0.45, 0.45), (1.00, 0.97, 0.97)),
            'blue' :  ( (0.0, 1.00, 1.00), (0.02, 0.75, 0.75), (1.00, 0.45, 0.45)) }
        
        cm = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
        '''
        #cm = self.__my_colormap()
        #cmap = cm.viridis(np.linspace(0, 1, int(grid.max()-grid.min())+1))
        
        heatmap = plt.pcolor(grid, cmap=cm.viridis)#, vmin=-40, vmax=40)
        
        plt.colorbar(heatmap)
        plt.title(title)
        
        plt.savefig(filename)


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
        
        nrows, ncols = grid.shape
        
        # resize
        figsize = self.__calculate_figsize(nrows, ncols)
        plt.figure(figsize=figsize, dpi=96)
        
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

        frame1.axes.xaxis.set_visible(False)
        frame1.axes.yaxis.set_visible(False)
        '''
        cdict = {
            'red'  :  ( (0.0, 0.25, 0.25), (0.02, 0.59, 0.59), (1.00, 1.00, 1.00)),
            'green':  ( (0.0, 0.00, 0.00), (0.02, 0.45, 0.45), (1.00, 0.97, 0.97)),
            'blue' :  ( (0.0, 1.00, 1.00), (0.02, 0.75, 0.75), (1.00, 0.45, 0.45)) }
        
        cm = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
        '''
        heatmap = plt.pcolor(grid, cmap=cm.viridis)#, vmin=-40, vmax=40)
        
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
        

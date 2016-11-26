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
        
        cmap = colors.ListedColormap(['white', 'green', 'black', 'red', 'blue', 'brown', 'gold'])
        
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

    def plot_results(self, grid, value, policy, title, filename):
        plt.figure(figsize=(12, 6))
        plt.title(title)
        #plt.xlabel(xlabel)
        #plt.ylabel(ylabel)
        c = plt.pcolor(data, edgecolors='k', linewidths=4, cmap='RdBu', vmin=0.0, vmax=1.0)
        
        #def show_values(pc, fmt="%.2f", **kw):
        #    from itertools import izip
        c.update_scalarmappable()
        ax = c.get_axes()
        for p, color, value in izip(c.get_paths(), c.get_facecolors(), c.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color)
    
        #show_values(c)
        
        plt.colorbar(c)


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
        plt.colorbar(heatmap)
        plt.title(title)
        
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                '''
                self.space_end = 2
                self.space_void = 3
                self.space_quicksand = 4
                '''
                
                if grid_labels[y, x] == 1:
                    # start
                    txt = '%.2f' % grid[y, x] + '\nS ' + self.get_arrow(grid_policy, x, y)
                    print(str(x), str(y), txt)
                    
                    plt.text(y + 0.5, x + 0.5, txt,
                             horizontalalignment='center',
                             verticalalignment='center')
                    
                elif grid_labels[y, x] == 4:
                    # quicksand
                    txt = '%.2f' % grid[y, x] + '\nQ ' + self.get_arrow(grid_policy, x, y)
                    print(str(x), str(y), txt)
                    plt.text(y + 0.5, x + 0.5, txt,
                             horizontalalignment='center',
                             verticalalignment='center')
                    
                elif grid_labels[y, x] == 3:
                    # obstacle
                    txt = '%.2f' % grid[y, x] + '\nX ' + self.get_arrow(grid_policy, x, y)
                    print(str(x), str(y), txt)
                    plt.text(y + 0.5, x + 0.5, txt,
                             horizontalalignment='center',
                             verticalalignment='center')
                    
                elif grid_labels[y, x] == 2:
                    # obstacle
                    txt = '%.2f' % grid[y, x] + '\nG ' + self.get_arrow(grid_policy, x, y)
                    print(str(x), str(y), txt)
                    plt.text(y + 0.5, x + 0.5, txt,
                             horizontalalignment='center',
                             verticalalignment='center')
        
                else:
                    txt = '%.2f' % grid[y, x] + '\n  ' + self.get_arrow(grid_policy, x, y)
                    print(str(x), str(y), txt)
                    plt.text(y + 0.5, x + 0.5, txt,
                             horizontalalignment='center',
                             verticalalignment='center')
        
        
        

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
        
    ##
    ## below is from stackoverflow
    ##
    def show_values(self, pc, grid, policy, fmt="%.2f", **kw):
        '''
        Heatmap with text in each cell with matplotlib's pyplot
        Source: http://stackoverflow.com/a/25074150/395857 
        By HYRY
        '''
        
        pc.update_scalarmappable()
        ax = pc.get_axes()
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            print(str(grid[(x,y)]))
            print(str(policy[(x,y)]))
            print(str(value))
            if grid[(y, x)] == 1:
                # start
                txt = '%.2f' % value + '\nS ' + self.get_arrow(policy, x, y)
                print(str(x), str(y), txt)
            elif grid[(y, x)] == 4:
                # quicksand
                txt = '%.2f' % value + '\nQ ' + self.get_arrow(policy, x, y)
                print(str(x), str(y), txt)
                
            elif grid[(y, x)] == 3:
                # obstacle
                txt = '%.2f' % grid[(x, y)] + '\nX ' + self.get_arrow(policy, x, y)
                print(str(x), str(y), txt)
                
            elif grid[(y, x)] == 2:
                # obstacle
                txt = '%.2f' % value + '\nG ' + self.get_arrow(policy, x, y)
                print(str(x), str(y), txt)
                
            elif grid[(y, x)] == 5:
                # bonus
                txt = '%.2f' % value + '\nB ' + self.get_arrow(policy, x, y)
                print(str(x), str(y), txt)
                
            else:
                txt = '%.2f' % value + '\n  ' + self.get_arrow(policy, x, y)
                print(str(x), str(y), txt)

            #ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)
            ax.text(x, y, txt, ha="center", va="center", color=color, **kw)
    
    def cm2inch(self, *tupl):
        '''
        Specify figure size in centimeter in matplotlib
        Source: http://stackoverflow.com/a/22787457/395857
        By gns-ank
        '''
        inch = 2.54
        if type(tupl[0]) == tuple:
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)
    
    def heatmap(self, value, grid, policy, title, xticklabels, yticklabels):
        '''
        Inspired by:
        - http://stackoverflow.com/a/16124677/395857 
        - http://stackoverflow.com/a/25074150/395857
        '''
    
        # Plot it out
        fig, ax = plt.subplots()    
        c = ax.pcolor(value, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu_r', vmin=value.min(), vmax=value.max())
    
        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(value.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(value.shape[1]) + 0.5, minor=False)
    
        # set tick labels
        #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)
    
        # set title and x/y labels
        plt.title(title)
        #plt.xlabel(xlabel)
        #plt.ylabel(ylabel)      
    
        # Remove last blank column
        plt.xlim( (0, value.shape[1]) )
    
        # Turn off all the ticks
        ax = plt.gca()    
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
    
        # Add color bar
        plt.colorbar(c)
    
        # Add text in each cell 
        self.show_values(c, grid, policy)
    
        # resize 
        fig = plt.gcf()
        fig.set_size_inches(self.cm2inch(30, 20))
    
    
    
    def plot_results2(self, value, grid,  policy, title, filename):
        plt.clf()
        plt.cla()
        plt.close('all')
        
        value = value[::-1]
        grid = grid[::-1]
        policy = policy[::-1]
        
        x_axis_size = grid.shape[0]
        y_axis_size = grid.shape[1]
        #title = "ROC's AUC"
        #xlabel= "Timeshift"
        #ylabel="Scales"
        #data =  np.random.rand(y_axis_size,x_axis_size)
        xticklabels = range(1, x_axis_size+1) # could be text
        yticklabels = range(1, y_axis_size+1) # could be text   
        self.heatmap(value, grid, policy, title, xticklabels, yticklabels)
        #plt.title(title)
        
        plt.savefig(filename, dpi=600, format='png', bbox_inches='tight')
        
        #plt.savefig('image_output.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
        #plt.show()
            

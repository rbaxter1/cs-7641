import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import itertools
import matplotlib.cm as cm

import matplotlib.ticker as ticker

from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import linalg
#import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.cluster import KMeans
from data_helper import *
from sklearn.datasets import make_blobs
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture, GMM

def run_and_plot():
    dh = data_helper()
    #X_train, X_test, y_train, y_test = dh.load_preprocess_and_split_titanic_data()
    X_train, X_test, y_train, y_test = dh.load_raw_titanic_data()
    y_pred = KMeans(n_clusters=4).fit_predict(X_train[:,1:3])
    print(y_pred)
    plt.scatter(X_train[:, 1], X_train[:, 2], c=y_pred)
    plt.show()
        

    plt.figure(figsize=(12, 12))
    
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # Incorrect number of clusters
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)
    
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Incorrect Number of Blobs")
    plt.show()
    
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y, means, covariances, index, title, fn):
    splot = plt.subplot(5, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.savefig(fn)


def nba_clusters():
    df = pd.read_csv('./data/shot_logs.csv', sep=',')
    
    le = LabelEncoder()
    le.fit(df['LOCATION'])
    le.transform(df['LOCATION']) 
    df['LOCATION_ENC'] = le.transform(df['LOCATION'])
        
    le = LabelEncoder()
    le.fit(df['SHOT_RESULT'])
    le.transform(df['SHOT_RESULT']) 
    df['SHOT_RESULT_ENC'] = le.transform(df['SHOT_RESULT'])
    
    x_col_names = ['SHOT_DIST', 'TOUCH_TIME', 'CLOSE_DEF_DIST', 'DRIBBLES']
    df.dropna(how='all', inplace=True)
    
    x = df.loc[:,x_col_names].values
    y = df.loc[:,'SHOT_RESULT_ENC'].values
    
    # split the data into training and test data
    # for the wine data using 30% of the data for testing
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.70,
                                                        random_state=0)
    df = pd.DataFrame(X_train)
    df.columns = x_col_names
    
    gen_plots(df, 'output_clustering_nba')
    

def wine_clusters():
    '''   
    CORR
                           quality  
    fixed acidity        0.124052  
    volatile acidity     -0.390558  
    citric acid          0.226373  
    residual sugar       0.013732  
    chlorides            -0.128907  
    free sulfur dioxide  -0.050656  
    total sulfur dioxide -0.185100  
    density              -0.174919  
    pH                    -0.057731  
    sulphates             0.251397  
    alcohol               0.476166  
    quality               1.000000  
    '''
    
    df = pd.read_csv('./data/winequality-red.csv', sep=';')
    
    split = df['quality'].median()
    df['quality_2'] = df['quality']
    
    # group the quality into binary good or bad
    df.loc[(df['quality'] >= 0) & (df['quality'] < split), 'quality_2'] = 0
    df.loc[(df['quality'] >= split), 'quality_2'] = 1
    
    df.dropna(how='all', inplace=True)
    
    x_col_names = ['alcohol', 'volatile acidity', 'sulphates', 'pH'] 
    
    x = df.loc[:,x_col_names].values
    y = df.loc[:,'quality_2'].values
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    
    df = pd.DataFrame(X_train)
    df.columns = x_col_names
    
    #gen_plots(df, 'output_clustering_wine')
    
    gen_all_plots(df, 'output_clustering_wine', 'Wine')
    
def gen_plots(df, out_dir):
    
    for i in range(df.shape[1]):
        
        for j in range(df.shape[1]):
        
            if i == j:
                continue
            
            f1 = df.columns[i]
            f2 = df.columns[j]
            
            print('Feature1: ', f1, ', Feature2: ', f2)
            X_train = df.values[:,(i,j)]
            #X_train_scale = StandardScaler().fit_transform(X_train)
            X_train_minmax = MinMaxScaler().fit_transform(X_train)
            
            km_inertias = []
            em_bic = []
            em_aic = []
            
            km_sil_score = []
            em_sil_score = []
            
            cluster_range = np.arange(2, 20, 1)
            for k in cluster_range:
                print('K Clusters: ', k)
                
                ##
                ## k-means
                ##
                km = KMeans(n_clusters=k, algorithm='full')
                km.fit(X_train_minmax)
                y_pred = km.predict(X_train_minmax)
                # inertia is the sum of distances from each point to its center   
                km_inertias.append(km.inertia_)
                km_sil_score.append(silhouette_score(X_train_minmax, y_pred, metric='euclidean'))
                #km_sil_score.append(1)

                # Clusters plot
                plt.clf()
                plt.cla()
                plt.scatter(X_train_minmax[:,0], X_train_minmax[:,1], c=y_pred)
                
                t = 'K-means clusters: ' + f1 + ' vs ' + f2 + ', k=' + str(k)
                plt.title(t)
                plt.xlabel(f1)
                plt.ylabel(f2)
                
                fn = './' + out_dir + '/' + f1 + '_' + f2 + '_' + str(k) + '_km_clusters.png'
                plt.savefig(fn)
                plt.close('all')
                
                ##
                ## Expectation Maximization
                ##
                em = GaussianMixture(n_components=k, covariance_type='full')
                em.fit(X_train_minmax)
                y_pred = em.predict(X_train_minmax)
                             
                em_bic.append(em.bic(X_train_minmax))
                em_aic.append(em.aic(X_train_minmax))
                em_sil_score.append(silhouette_score(X_train_minmax, y_pred, metric='euclidean'))
                #em_sil_score.append(1)

                # Clusters plot
                plt.clf()
                plt.cla()
                plt.scatter(X_train_minmax[:,0], X_train_minmax[:,1], c=y_pred)
                
                t = 'EM clusters: ' + f1 + ' vs ' + f2 + ', k=' + str(k)
                plt.title(t)
                plt.xlabel(f1)
                plt.ylabel(f2)
                
                fn = './' + out_dir + '/' + f1 + '_' + f2 + '_' + str(k) + '_em_clusters.png'
                plt.savefig(fn)
                plt.close('all')
                
                
            # K-means Elbow plot
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots()
            
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_locator(ticker.MaxNLocator(integer=True))
                
            plt.plot(cluster_range, km_inertias)
            
            t = 'K-Means Elbow: ' + f1 + ' vs ' + f2
            plt.title(t)
            
            
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            
            fn = './' + out_dir + '/' + f1 + '_' + f2 + '_km_elbow.png'
            plt.savefig(fn)
            plt.close('all')
            
            
            # K-means Silhouette plot
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots()
            
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_locator(ticker.MaxNLocator(integer=True))
                
            plt.plot(cluster_range, km_sil_score)
            
            t = 'K-Means Silhouette: ' + f1 + ' vs ' + f2
            plt.title(t)
            
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette')
            
            fn = './' + out_dir + '/' + f1 + '_' + f2 + '_km_silhouette.png'
            plt.savefig(fn)
            plt.close('all')
            
            
            # EM Silhouette plot
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots()
            
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_locator(ticker.MaxNLocator(integer=True))
                
            plt.plot(cluster_range, em_sil_score)
            
            t = 'EM Silhouette: ' + f1 + ' vs ' + f2
            plt.title(t)
            
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette')
            
            fn = './' + out_dir + '/' + f1 + '_' + f2 + '_em_silhouette.png'
            plt.savefig(fn)
            plt.close('all')
            
            
            # EM BIC/AIC plot
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots()
            
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            plt.plot(cluster_range, em_bic, label='BIC')
            plt.plot(cluster_range, em_aic, label='AIC')
                
            t = 'EM IC: ' + f1 + ' vs ' + f2
            plt.title(t)
            
            plt.xlabel('Number of Clusters')
            plt.ylabel('Information Criterion')
            
            plt.legend(loc='best')

            fn = './' + out_dir + '/' + f1 + '_' + f2 + '_em_ic.png'
            plt.savefig(fn)
            plt.close('all')
            
            print('done ', f1, ', ', f2)
    
    
def gen_all_plots(df, out_dir, name):
    X_train = df.values
    X_train_minmax = MinMaxScaler().fit_transform(X_train)
    
    km_inertias = []
    em_bic = []
    em_aic = []
    
    km_sil_score = []
    em_sil_score = []
    
    cluster_range = np.arange(2, 20, 1)
    for k in cluster_range:
        print('K Clusters: ', k)
        
        ##
        ## k-means
        ##
        km = KMeans(n_clusters=k, algorithm='full')
        km.fit(X_train_minmax)
        y_pred = km.predict(X_train_minmax)
        # inertia is the sum of distances from each point to its center   
        km_inertias.append(km.inertia_)
        km_sil_score.append(silhouette_score(X_train_minmax, y_pred, metric='euclidean'))
        #km_sil_score.append(1)

        ##
        ## Expectation Maximization
        ##
        em = GaussianMixture(n_components=k, covariance_type='full')
        em.fit(X_train_minmax)
        y_pred = em.predict(X_train_minmax)
                     
        em_bic.append(em.bic(X_train_minmax))
        em_aic.append(em.aic(X_train_minmax))
        em_sil_score.append(silhouette_score(X_train_minmax, y_pred, metric='euclidean'))
        #em_sil_score.append(1)
        
        
    # K-means Elbow plot
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
        
    plt.plot(cluster_range, km_inertias)
    
    t = 'K-Means Elbow: ' + name
    plt.title(t)
    
    
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    
    fn = './' + out_dir + '/' + name + '_km_elbow.png'
    plt.savefig(fn)
    plt.close('all')
    
    
    # K-means Silhouette plot
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
        
    plt.plot(cluster_range, km_sil_score)
    
    t = 'K-Means Silhouette: ' + name
    plt.title(t)
    
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette')
    
    fn = './' + out_dir + '/' + name + '_km_silhouette.png'
    plt.savefig(fn)
    plt.close('all')
    
    
    # EM Silhouette plot
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
        
    plt.plot(cluster_range, em_sil_score)
    
    t = 'EM Silhouette: ' + name
    plt.title(t)
    
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette')
    
    fn = './' + out_dir + '/' + name + '_em_silhouette.png'
    plt.savefig(fn)
    plt.close('all')
    
    
    # EM BIC/AIC plot
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.plot(cluster_range, em_bic, label='BIC')
    plt.plot(cluster_range, em_aic, label='AIC')
        
    t = 'EM IC: ' + f1 + ' vs ' + f2
    plt.title(t)
    
    plt.xlabel('Number of Clusters')
    plt.ylabel('Information Criterion')
    
    plt.legend(loc='best')

    fn = './' + out_dir + '/' + name + '_em_ic.png'
    plt.savefig(fn)
    plt.close('all')
    
    print('done ', name)


if __name__== '__main__':
    wine_clusters()
    
    

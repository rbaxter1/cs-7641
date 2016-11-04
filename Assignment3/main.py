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
import matplotlib as mpl

from sklearn.cluster import KMeans
from data_helper import *
from sklearn.datasets import make_blobs
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture, GMM
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
    
    df = df.drop(df[df['TOUCH_TIME'] < 0].index)
    
    x_col_names = ['SHOT_DIST', 'TOUCH_TIME', 'CLOSE_DEF_DIST', 'DRIBBLES']
    df = df.dropna(how='all', inplace=True)
    
    x = df.loc[:,x_col_names].values
    y = df.loc[:,'SHOT_RESULT_ENC'].values
    
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.30,
                                                        random_state=0)
    df = pd.DataFrame(X_train)
    df.columns = x_col_names
    
    gen_cluster_plots(df, 'output_clustering_nba', 40)
    gen_cluster_all_plots(df, 'output_clustering_nba', 'NBA', 40)
    

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
    
    gen_cluster_plots(df, 'output_clustering_wine', 20)
    gen_cluster_all_plots(df, 'output_clustering_wine', 'Wine', 20)
    
    
def nba_dim_reduce():
    
    df = pd.read_csv('./data/shot_logs.csv', sep=',')
        
    le = LabelEncoder()
    le.fit(df['LOCATION'])
    le.transform(df['LOCATION']) 
    df['LOCATION_ENC'] = le.transform(df['LOCATION'])
        
    le = LabelEncoder()
    le.fit(df['SHOT_RESULT'])
    le.transform(df['SHOT_RESULT']) 
    df['SHOT_RESULT_ENC'] = le.transform(df['SHOT_RESULT'])
    
    df = df.drop(df[df['TOUCH_TIME'] < 0].index)
    
    x_col_names = ['SHOT_DIST', 'TOUCH_TIME', 'LOCATION_ENC', 'PTS_TYPE', 'DRIBBLES', 'CLOSE_DEF_DIST']
    
    y = df.loc[:,'SHOT_RESULT_ENC'].values
    df = df.drop('SHOT_RESULT_ENC', 1)
    x = df.loc[:,x_col_names].values
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    
    dfx = pd.DataFrame(X_train)
    dfx.columns = x_col_names
    
    dfy = pd.DataFrame(y_train)
    dfy.columns = ['SHOT_RESULT_ENC']
    
    dfxt = pd.DataFrame(X_test)
    dfxt.columns = x_col_names
    
    dfyt = pd.DataFrame(y_test)
    dfyt.columns = ['SHOT_RESULT_ENC']
    
    gen_dim_reduce_plots(dfx, dfy, dfxt, dfyt, 'output_dim_reduce_nba', 'NBA', dfx.shape[1])
    
def wine_dim_reduce():
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
    
    df.dropna(how='all', inplace=True)
    
    split = df['quality'].median()
    df['quality_2'] = df['quality']
    
    # group the quality into binary good or bad
    df.loc[(df['quality'] >= 0) & (df['quality'] < split), 'quality_2'] = 0
    df.loc[(df['quality'] >= split), 'quality_2'] = 1
    
    
    x_col_names = ['fixed acidity', 'citric acid', 'alcohol', 'residual sugar', 'chlorides', 'volatile acidity', 'sulphates', 'pH'] 

    y = df.loc[:,'quality_2'].values
    df = df.drop('quality', 1)
    df = df.drop('quality_2', 1)
    x = df.values
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    
    dfx = pd.DataFrame(X_train)
    dfx.columns = df.columns
    
    dfy = pd.DataFrame(y_train)
    dfy.columns = ['quality']
    
    dfxt = pd.DataFrame(X_test)
    dfxt.columns = df.columns
    
    dfyt = pd.DataFrame(y_test)
    dfyt.columns = ['quality']
    
    gen_dim_reduce_plots(dfx, dfy, dfxt, dfyt, 'output_dim_reduce_wine', 'Wine', dfx.shape[1])
    
    
    
def gen_cluster_plots(df, out_dir, max_clusters):
    
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
            
            cluster_range = np.arange(2, max_clusters, 1)
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
                #km_sil_score.append(silhouette_score(X_train_minmax, y_pred, metric='euclidean'))
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
                #em_sil_score.append(silhouette_score(X_train_minmax, y_pred, metric='euclidean'))
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
            '''
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
            '''
            
            
            # EM Silhouette plot
            '''
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
            '''
            
            
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
    
    

def gen_dim_reduce_plots(dfx, dfy, dfxt, dfyt, out_dir, name, max_clusters):
    X_train = dfx.values
    #X_train_scale = StandardScaler().fit_transform(X_train)
    X_train_minmax = MinMaxScaler().fit_transform(X_train)
    
    y_train = dfy.values
    #X_train_scale = StandardScaler().fit_transform(X_train)
    y_train_minmax = MinMaxScaler().fit_transform(y_train)
    
    X_test = dfxt.values
    X_test_minmax = MinMaxScaler().fit_transform(X_test)
    
    y_test = dfyt.values
    #X_train_scale = StandardScaler().fit_transform(X_train)
    y_test_minmax = MinMaxScaler().fit_transform(y_test)
    
    
    #explained_var_ratio = []
    
    #cluster_range = np.arange(2, max_clusters, 1)
    #for k in cluster_range:
    #print('K: ', k)
    
    ##
    ## PCA
    ##
    pca = PCA(n_components=dfx.shape[1], svd_solver='full')
    pca.fit(X_train_minmax)
    X_train_pca = pca.transform(X_train_minmax)
    #explained_var_ratio.append(pca.explained_variance_ratio_)
    print(X_train_pca.shape)
    
    # Scatter plot
    '''        
    plt.clf()
    plt.cla()
    plt.scatter(X_train_pca[:,0], X_train_pca[:,1])
    
    t = 'PCA Scatter: ' + f1 + ' vs ' + f2 + ', k=' + str(k)
    plt.title(t)
    plt.xlabel(f1)
    plt.ylabel(f2)
    
    fn = './' + out_dir + '/' + f1 + '_' + f2 + '_' + str(k) + '_pca_scatter.png'
    plt.savefig(fn)
    plt.close('all')
    '''
        
        
    # PCA plot
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    
    #for axis in [ax.xaxis, ax.yaxis]:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
    rng = np.arange(1, pca.explained_variance_ratio_.shape[0]+1, 1)
    
    plt.bar(rng, pca.explained_variance_ratio_,
            alpha=0.5, align='center',
            label='Individual Explained Variance')
    
    plt.step(rng, np.cumsum(pca.explained_variance_ratio_),
            where='mid', label='Cumulative Explained Variance')
    
    plt.legend(loc='best')
    
    t = 'PCA: ' + name
    plt.title(t)
    
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    
    fn = './' + out_dir + '/' + name + '_pca_evr.png'
    plt.savefig(fn)
    plt.close('all')
    
    
    
    ##
    ## ICA
    ##
    ica = FastICA(n_components=dfx.shape[1])
    ica.fit(X_train_minmax)
    X_train_ica = ica.transform(X_train_minmax)
    print(X_train_ica.shape)
    #explained_var_ratio.append(pca.explained_variance_ratio_)
    
    '''
    # ICA plot
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    
    #for axis in [ax.xaxis, ax.yaxis]:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    #rng = np.arange(1, ica.explained_variance_ratio_.shape[0]+1, 1)
    
    plt.bar(rng, pca.explained_variance_ratio_,
            alpha=0.5, align='center',
            label='Individual Explained Variance')
    
    plt.step(rng, np.cumsum(pca.explained_variance_ratio_),
            where='mid', label='Cumulative Explained Variance')
    
    plt.legend(loc='best')
    
    t = 'ICA: ' + name
    plt.title(t)
    
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    
    fn = './' + out_dir + '/' + name + '_ica_evr.png'
    plt.savefig(fn)
    plt.close('all')
    '''
    

    ##
    ## Random Projection
    ##
    rp = GaussianRandomProjection(n_components=dfx.shape[1])
    rp.fit(X_train_minmax)
    X_train_rp = rp.transform(X_train_minmax)
    print(X_train_rp.shape)
    
    
    
    
    ##
    ## LDA
    ##
    lda = LinearDiscriminantAnalysis(n_components=dfx.shape[1])
    lda.fit(X_train_minmax, y_train_minmax)
    
    score_ = lda.score(X_test_minmax, y_test_minmax)
    
    X_train_lda = lda.transform(X_train_minmax)
    print(X_train_lda.shape)
    #X_train_lda.shape
    
    # LDA plot
    
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    
    #for axis in [ax.xaxis, ax.yaxis]:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
    rng = np.arange(1, lda.explained_variance_ratio_.shape[0]+1, 1)
    
    plt.bar(rng, lda.explained_variance_ratio_,
            alpha=0.5, align='center',
            label='Individual Explained Variance')
    
    plt.step(rng, np.cumsum(lda.explained_variance_ratio_),
            where='mid', label='Cumulative Explained Variance')
    
    plt.legend(loc='best')
    
    t = 'LDA: ' + name
    plt.title(t)
    
    plt.xlabel('Components')
    plt.ylabel('Explained Variance Ratio')
    
    fn = './' + out_dir + '/' + name + '_lda_evr.png'
    plt.savefig(fn)
    plt.close('all')
    
    
    print('done ', name)
    
    
    
def gen_cluster_all_plots(df, out_dir, name, max_clusters):
    X_train = df.values
    X_train_minmax = MinMaxScaler().fit_transform(X_train)
    
    km_inertias = []
    em_bic = []
    em_aic = []
    
    km_sil_score = []
    em_sil_score = []
    
    cluster_range = np.arange(2, max_clusters, 1)
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
        #km_sil_score.append(silhouette_score(X_train_minmax, y_pred, metric='euclidean'))
        #km_sil_score.append(1)

        ##
        ## Expectation Maximization
        ##
        em = GaussianMixture(n_components=k, covariance_type='full')
        em.fit(X_train_minmax)
        y_pred = em.predict(X_train_minmax)
                     
        em_bic.append(em.bic(X_train_minmax))
        em_aic.append(em.aic(X_train_minmax))
        #em_sil_score.append(silhouette_score(X_train_minmax, y_pred, metric='euclidean'))
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
    '''
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
    '''
    
    
    # EM Silhouette plot
    '''
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
    '''
    
    # EM BIC/AIC plot
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.plot(cluster_range, em_bic, label='BIC')
    plt.plot(cluster_range, em_aic, label='AIC')
        
    t = 'EM IC: ' + name
    plt.title(t)
    
    plt.xlabel('Number of Clusters')
    plt.ylabel('Information Criterion')
    
    plt.legend(loc='best')

    fn = './' + out_dir + '/' + name + '_em_ic.png'
    plt.savefig(fn)
    plt.close('all')
    
    print('done ', name)


if __name__== '__main__':
    #wine_clusters()
    #nba_clusters()
    wine_dim_reduce()
    nba_dim_reduce()
    

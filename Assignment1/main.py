import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Imputer

import io
import pydotplus

from tree_test import *
from knn_test import *
from svm_test import *
from boost_test import *
from neural_test import *

from timeit import default_timer as timer


def main2(runTree=True, runKnn=True, runSvm=True, runBoost=True, runNeural=True, runWine=True, runTitanic=True, ):
    
    
    #
    #
    # RED WINE
    #
    #    
    if runWine:
            
        # load the red wine data
        # source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
        df = pd.read_csv('./data/winequality-red.csv', sep=';')
        
        # group the quality into binary good or bad
        df.loc[(df['quality'] >= 0) & (df['quality'] <= 5), 'quality'] = 0
        df.loc[(df['quality'] >= 6), 'quality'] = 100
        
        # separate the x and y data
        # y = quality, x = features (using fixed acid, volatile acid and alcohol)
        x_col_names = ['fixed acidity', 'volatile acidity', 'alcohol']
        x, y = df.loc[:,x_col_names].values, df.loc[:,'quality'].values
        
        # split the data into training and test data
        # for the wine data using 30% of the data for testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        
        #
        # TREE
        #
        if runTree:
            myModel = rb_tree_test(x_train, x_test, y_train, y_test, x_col_names, 'redwine_tree', cv=5)
            
            start = timer()
            myModel.run_model(max_depth=4, criterion='entropy')
            end = timer()
            print('redwine_tree run_model took:', end - start)
            
            start = timer()
            myModel.run_cv_model(max_depth=4, criterion='entropy')
            end = timer()
            print('redwine_tree run_cv_model took:', end - start)
            
            start = timer()
            myModel.plot_validation_curve(max_depth=4, criterion='entropy')
            end = timer()
            print('redwine_tree plot_validation_curve took:', end - start)
            
        
        
        #
        # KNN
        #
        if runKnn:
            myModel = rb_knn_test(x_train, x_test, y_train, y_test, x_col_names, 'redwine_knn', cv=5)
            
            start = timer()
            myModel.run_model(n_neighbors=20, leaf_size=30, p=5)
            end = timer()
            print('redwine_knn run_model took:', end - start)
            
            start = timer()
            myModel.run_cv_model(n_neighbors=20, leaf_size=30, p=5)
            end = timer()
            print('redwine_knn run_cv_model took:', end - start)
            
            start = timer()
            myModel.plot_validation_curve(n_neighbors=20, leaf_size=30, p=5)
            end = timer()
            print('redwine_knn plot_validation_curve took:', end - start)
        
        
        #
        # SVM
        #
        if runSvm:
            myModel = rb_svm_test(x_train, x_test, y_train, y_test, x_col_names, 'redwine_svm', cv=5)
            
            start = timer()
            myModel.run_model(C=4.0, degree=3, cache_size=200)
            end = timer()
            print('redwine_svm run_model took:', end - start)
            
            start = timer()
            myModel.run_cv_model(C=4.0, degree=3, cache_size=200)
            end = timer()
            print('redwine_svm run_cv_model took:', end - start)
            
            start = timer()
            myModel.plot_validation_curve(C=4.0, degree=3, cache_size=200)
            end = timer()
            print('redwine_svm plot_validation_curve took:', end - start)
            
            
        #
        # Boost
        #
        if runBoost:
            myModel = rb_boost_test(x_train, x_test, y_train, y_test, x_col_names, 'redwine_boost', cv=5)
            
            start = timer()
            myModel.run_model(max_depth=1, criterion='entropy', learning_rate=1., n_estimators=300)
            end = timer()
            print('redwine_boost run_model took:', end - start)
            
            start = timer()
            myModel.run_cv_model(max_depth=1, criterion='entropy', learning_rate=1., n_estimators=300)
            end = timer()
            print('redwine_boost run_cv_model took:', end - start)
            
            start = timer()
            myModel.plot_validation_curve(max_depth=1, criterion='entropy', learning_rate=1., n_estimators=300)
            end = timer()
            print('redwine_boost plot_validation_curve took:', end - start)
        
        
        #
        # Neural
        #
        if runNeural:
            myModel = rb_neural_test(x_train, x_test, y_train, y_test, x_col_names, 'redwine_neural', cv=5)
            
            start = timer()
            myModel.run_model(alpha=0.0001, batch_size=200, learning_rate_init=0.001, power_t=0.5, max_iter=200, momentum=0.9, beta_1=0.9, beta_2=0.999, hidden_layer_sizes=(100,))
            end = timer()
            print('redwine_neural run_model took:', end - start)
            
            start = timer()
            myModel.run_cv_model(alpha=0.0001, batch_size=200, learning_rate_init=0.001, power_t=0.5, max_iter=200, momentum=0.9, beta_1=0.9, beta_2=0.999, hidden_layer_sizes=(100,))
            end = timer()
            print('redwine_neural run_cv_model took:', end - start)
            
            start = timer()
            myModel.plot_validation_curve(alpha=0.0001, batch_size=200, learning_rate_init=0.001, power_t=0.5, max_iter=200, momentum=0.9, beta_1=0.9, beta_2=0.999, hidden_layer_sizes=(100,))
            end = timer()
            print('redwine_neural plot_validation_curve took:', end - start)
            
            
            
            
    #
    #
    # TITANIC
    #
    #    
    if runTitanic:
        # source: https://www.kaggle.com/c/titanic/data
        df = pd.read_csv('./data/titanic_train.csv', sep=',')
        
        # we need to encode sex. using the sklearn label encoder is
        # one way. however one consideration is that the learning
        # algorithm may make assumptions about the magnitude of the
        # labels. for example, male is greater than female. use
        # one hot encoder to get around this.
        #ohe = OneHotEncoder(categorical_features=[0])
        #ohe.fit_transform(x).toarray()
        
        # Even better pandas has a one hot encoding built in!
        df = pd.get_dummies(df[['Sex', 'Pclass', 'Age', 'Survived']])    
    
        # this data set is missing some ages. we could impute a value
        # like the average or median. or remove the rows having missing
        # data. the disadvantage of removing values is we may be taking
        # away valuable information that the learning algorithm needs.
        imr = Imputer(strategy='most_frequent')
        imr.fit(df['Age'].reshape(-1, 1))
        imputed_data = imr.transform(df['Age'].reshape(-1, 1))
        
        df['Age']  = imputed_data
        
        y = df['Survived'].values
        x = df.iloc[:,[0,1,3,4]].values
        
        x_col_names = df.iloc[:,[0,1,3,4]].columns
        
        # split the data into training and test data
        # for the wine data using 30% of the data for testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        
        #y_train1 = y
        #x_train1 = x
        
        #df_test = pd.read_csv('./data/titanic_test.csv', sep=',')
        #df_test = pd.get_dummies(df_test[['Sex', 'Pclass', 'Age', 'Survived']])        

        #x_test1 = df_test.iloc[:,[0,1,3,4]].values
        
        
        
        #
        # TREE
        #
        if runTree:
            myModel = rb_tree_test(x_train, x_test, y_train, y_test, x_col_names, 'titanic_tree', cv=5)
            
            start = timer()
            myModel.run_model(max_depth=4, criterion='entropy')
            end = timer()
            print('titanic_tree run_model took:', end - start)
            
            start = timer()
            myModel.run_cv_model(max_depth=4, criterion='entropy')
            end = timer()
            print('titanic_tree run_cv_model took:', end - start)
            
            start = timer()
            myModel.plot_validation_curve(max_depth=4, criterion='entropy')
            end = timer()
            print('titanic_tree plot_validation_curve took:', end - start)
            
        
        
        #
        # KNN
        #
        if runKnn:
            myModel = rb_knn_test(x_train, x_test, y_train, y_test, x_col_names, 'titanic_knn', cv=5)
            
            start = timer()
            myModel.run_model(n_neighbors=20, leaf_size=30, p=4)
            end = timer()
            print('titanic_knn run_model took:', end - start)
            
            start = timer()
            myModel.run_cv_model(n_neighbors=20, leaf_size=30, p=4)
            end = timer()
            print('titanic_knn run_cv_model took:', end - start)
            
            start = timer()
            myModel.plot_validation_curve(n_neighbors=20, leaf_size=30, p=4)
            end = timer()
            print('titanic_knn plot_validation_curve took:', end - start)
        
        
        #
        # SVM
        #
        if runSvm:
            myModel = rb_svm_test(x_train, x_test, y_train, y_test, x_col_names, 'titanic_svm', cv=5)
            
            start = timer()
            myModel.run_model(C=2.0, degree=3, cache_size=200)
            end = timer()
            print('titanic_svm run_model took:', end - start)
            
            start = timer()
            myModel.run_cv_model(C=2.0, degree=3, cache_size=200)
            end = timer()
            print('titanic_svm run_cv_model took:', end - start)
            
            start = timer()
            myModel.plot_validation_curve(C=2.0, degree=3, cache_size=200)
            end = timer()
            print('titanic_svm plot_validation_curve took:', end - start)
            
            
        #
        # Boost
        #
        if runBoost:
            myModel = rb_boost_test(x_train, x_test, y_train, y_test, x_col_names, 'titanic_boost', cv=5)
            
            start = timer()
            myModel.run_model(max_depth=1, criterion='entropy', learning_rate=1., n_estimators=300)
            end = timer()
            print('titanic_boost run_model took:', end - start)
            
            start = timer()
            myModel.run_cv_model(max_depth=1, criterion='entropy', learning_rate=1., n_estimators=300)
            end = timer()
            print('titanic_boost run_cv_model took:', end - start)
            
            start = timer()
            myModel.plot_validation_curve(max_depth=1, criterion='entropy', learning_rate=1., n_estimators=300)
            end = timer()
            print('titanic_boost plot_validation_curve took:', end - start)
        
        
        #
        # Neural
        #
        if runNeural:
            myModel = rb_neural_test(x_train, x_test, y_train, y_test, x_col_names, 'titanic_neural', cv=5)
            
            start = timer()
            myModel.run_model(alpha=0.0001, batch_size=100, learning_rate_init=0.001, power_t=0.5, max_iter=200, momentum=0.9, beta_1=0.9, beta_2=0.999, hidden_layer_sizes=(100,))
            end = timer()
            print('titanic_neural run_model took:', end - start)
            
            start = timer()
            myModel.run_cv_model(alpha=0.0001, batch_size=100, learning_rate_init=0.001, power_t=0.5, max_iter=200, momentum=0.9, beta_1=0.9, beta_2=0.999, hidden_layer_sizes=(100,))
            end = timer()
            print('titanic_neural run_cv_model took:', end - start)
            
            start = timer()
            myModel.plot_validation_curve(alpha=0.0001, batch_size=100, learning_rate_init=0.001, power_t=0.5, max_iter=200, momentum=0.9, beta_1=0.9, beta_2=0.999, hidden_layer_sizes=(100,))
            end = timer()
            print('titanic_neural plot_validation_curve took:', end - start)
            
    
if __name__ == "__main__":
    main2()

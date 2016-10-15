from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, Imputer
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier
from plot_helper import *

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import host_subplot
#import mpl_toolkits.axisartist as AA

import numpy as np
import pandas as pd
from data_helper import *

from timeit import default_timer as timer

class validation_curves:
    def __init__(self):
        pass
    
    def gridSearch2(self):
        dh = data_helper()
        X_train, X_test, y_train, y_test =  dh.load_wine_data()

        pipeline = Pipeline([#('scl', StandardScaler()),
                             ('clf', DecisionTreeClassifier(random_state=0))])
        
        parameters = {'clf__criterion': ('gini', 'entropy'),
                      'clf__min_impurity_split': np.arange(0, 0.5, 0.01),
                      'clf__max_depth': np.arange(1, 40, 1)#,
                      #'clf__min_samples_split': np.arange(1, 200, 5),
                      #'clf__min_samples_leaf': np.arange(2, 200, 5),
                      #'clf__min_weight_fraction_leaf': np.arange(0.0, 0.5, 0.05),
                      #'clf__max_leaf_nodes': np.arange(2, 300, 5)
                      }
        
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)
        
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        print(parameters)
        t0 = timer()
        grid_search.fit(X_train, y_train)
        print("done in %0.3fs" % (timer() - t0))
        print()
    
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
            
    def run(self, X_train, X_test, y_train, y_test, clf_type, sc1_type, outer_param_dict, dataset, learner_name, complexity_name):
        
        ##'min_weight_fraction_leaf': np.arange(0.0, 0.5, 0.01),
        ph = plot_helper()
        for outer_param in outer_param_dict.keys():
            print(outer_param)
            
            for outer_param_value in outer_param_dict[outer_param].keys():
                print(outer_param_value)
                
                params_dict = outer_param_dict[outer_param][outer_param_value]
                    
                for param_name in params_dict.keys():
                    print(param_name)
                        
                    x = []
                    in_sample_avg_errors = []
                    std_in_sample_errors = []
                    out_of_sample_avg_errors = []
                    std_out_of_sample_errors = []
                    avg_complexity_measures = []
                    std_complexity_measures = []
                    avg_fit_times = []
                    std_fit_times = []
                    avg_predict_times = []
                    std_predict_times = []
                    
                    
                    for param_value in params_dict[param_name]['param_value']:
                        print(param_value)
                        
                        #clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
                        
                        if sc1_type == None:
                            clf = clf_type()
                        elif sc1_type == AdaBoostClassifier:
                            clf = AdaBoostClassifier(base_estimator=clf_type())
                        elif clf_type == MLPClassifier:
                            if dataset == 'Titanic':
                                clf = pipeline = Pipeline([('scl', StandardScaler()),
                                                           ('clf', MLPClassifier(random_state=0,
                                                                                 #max_iter=50,
                                                                                 activation='relu',
                                                                                 shuffle=True,
                                                                                 solver='adam',
                                                                                 learning_rate_init=0.001,
                                                                                 learning_rate='constant',
                                                                                 hidden_layer_sizes=(100,)
                                                                                 ))])
                                '''
                                clf = Pipeline([('scl', StandardScaler()),
                                              ('clf', MLPClassifier(random_state=0,
                                                                    #max_iter=50,
                                                                    activation='relu',
                                                                    shuffle=True,
                                                                    solver='adam',
                                                                    learning_rate_init=0.001,
                                                                    learning_rate='constant'
                                                                    ))])
                                '''
                            else:
                                clf = Pipeline([('scl', StandardScaler()),
                                          ('clf', MLPClassifier(activation='relu',
                                                                learning_rate='constant',
                                                                shuffle=True,
                                                                solver='adam',
                                                                random_state=0,
                                                                max_iter=500,
                                                                batch_size=60
                                                                ))])
                        elif clf_type == 'SVC':
                            if dataset == 'wine':
                                pipeline = Pipeline([('scl', StandardScaler()),
                                                     ('clf', SVC(C=1, random_state=0, max_iter=500, tol=0.001, kernel='poly'))])
                                
                            else:
                                pipeline = Pipeline([('scl', StandardScaler()),
                                                     ('clf', SVC(random_state=0, max_iter=500, tol=0.001,
                                                                 kernel='rbf', probability=True,
                                                                 shrinking=True, C=3))])
                                
                                
                        else:
                            clf = Pipeline([ ('scl', sc1_type() ),
                                             ('clf', clf_type() )
                                             ])        
                            
                        
                        params = clf.get_params()
                        params[param_name] = param_value
                        params[outer_param] = outer_param_value
                        clf.set_params(**params)
                        
                        # this is the 0.18dev version
                        skf = StratifiedKFold(n_splits=5, random_state=0)
                        
                        out_sample_errors = []
                        in_sample_errors = []
                        complexity_measures = []
                        fit_times = []
                        predict_times = []
                        
                        # do the cross validation
                        for k, (train, test) in enumerate(skf.split(X=X_train, y=y_train)):
                            
                            start = timer()
                            # run the learning algorithm
                            clf.fit(X_train[train], y_train[train])
                            end = timer()
                            tot_fit_time = (end - start) * 1000.
                            fit_times.append(tot_fit_time)
                            
                            # complexity
                            if complexity_name == 'Tree nodes':
                                complexity = clf.tree_.node_count
                            else:
                                complexity = 0
                                
                            complexity_measures.append(complexity)
                            
                            # in sample
                            start = timer()
                            predicted_values = clf.predict(X_train[train])
                            end = timer()
                            tot_in_sample_predict_time = (end - start) * 1000.
                            predict_times.append(tot_in_sample_predict_time)
                            
                            in_sample_mse = mean_squared_error(y_train[train], predicted_values)
                            in_sample_mse = 1. - clf.score(X_train[train], y_train[train])
                            
                            in_sample_errors.append(in_sample_mse)
                            
                            # out of sample
                            start = timer()
                            predicted_values = clf.predict(X_train[test])
                            end = timer()
                            tot_out_sample_predict_time = (end - start) * 1000.
                            predict_times.append(tot_out_sample_predict_time)
                            out_sample_mse = mean_squared_error(y_train[test], predicted_values)
                            out_sample_mse = 1. - clf.score(X_test, y_test)
                            out_sample_errors.append(out_sample_mse)
        
                            print('Fold:', k+1, ', Validation error: ', out_sample_mse,
                                  ', Training error: ', in_sample_mse, ', ', complexity_name, ': ', complexity,
                                  ', fit time: ', tot_fit_time, ', in sample predict time: ', tot_in_sample_predict_time,
                                  ', out of sample predict time: ', tot_out_sample_predict_time)
                           
                        # RUN WITH ALL THE DATA FOR PLOT
                        
                        '''
                        clf.fit(X_train, y_train)
                        predicted_values = clf.predict(X_test)
                        
                        
                        ph.plot_pred_act(y_test,
                                        predicted_values,
                                        dataset + '_' + learner_name + '_' + outer_param + '_' + outer_param_value + '_' + param_name,
                                        'Predicted versus Actual', 
                                        dataset, 
                                        save_file_name=dataset + '_' + learner_name + '_' + outer_param + '_' + outer_param_value + '_' + param_name)
                        '''
                        
                        scores = cross_val_score(clf, X_train, y_train)
                        out_sample_errors = 1. - scores
                        
                        # out of sample (test)
                        avg_test_err = np.mean(out_sample_errors)
                        test_err_std = np.std(out_sample_errors)
                        out_of_sample_avg_errors.append(avg_test_err)
                        std_out_of_sample_errors.append(test_err_std)
                        
                        # in sample (train)
                        avg_train_err = np.mean(in_sample_errors)
                        train_err_std = np.std(in_sample_errors)
                        in_sample_avg_errors.append(avg_train_err)
                        std_in_sample_errors.append(train_err_std)
                        
                        # timings
                        avg_fit_time = np.mean(fit_times)
                        std_fit_time = np.std(fit_times)
                        avg_fit_times.append(avg_fit_time)
                        std_fit_times.append(std_fit_time)
                        
                        avg_predict_time = np.mean(predict_times)
                        std_predict_time = np.std(predict_times)
                        avg_predict_times.append(avg_predict_time)
                        std_predict_times.append(std_predict_time)
                        
                        # complexity (num nodes) - how can this be universally applied?
                        avg_complexity = np.mean(complexity_measures)
                        std_complexity = np.std(complexity_measures)
                        avg_complexity_measures.append(avg_complexity)
                        std_complexity_measures.append(std_complexity)
                        
                        print('Avg validation MSE: ', avg_test_err, '+/-', test_err_std,
                              'Avg training MSE: ', avg_train_err, '+/-', train_err_std,
                              'Avg complexity: ', avg_complexity, '+/-', std_complexity,
                              'Avg fit time: ', avg_fit_time, '+/-', std_fit_time,
                              'Avg predict time: ', avg_predict_time, '+/-', std_predict_time)
                        
                        x.append(param_value)
                        
                    if clf_type == MLPClassifier:
                        x = np.arange(1, len(x)+1, 1)
                        
                    ph.plot_validation_curve(param_range=x,
                                             train_mean=np.array(in_sample_avg_errors),
                                             train_std=np.array(std_in_sample_errors),
                                             test_mean=np.array(out_of_sample_avg_errors),
                                             test_std=np.array(std_out_of_sample_errors),
                                             complexity_mean=np.array(avg_complexity_measures),
                                             complexity_std=np.array(std_complexity_measures),
                                             fit_time_mean=np.array(avg_fit_times),
                                             fit_time_std=np.array(std_fit_times),
                                             predict_time_mean=np.array(avg_predict_times),
                                             predict_time_std=np.array(std_predict_times),
                                             rev_axis=params_dict[param_name]['reverse_xaxis'],
                                             param_name=param_name,
                                             learner_name=learner_name,
                                             save_file_name=dataset + '_' + learner_name + '_' + outer_param + '_' + outer_param_value + '_' + param_name,
                                             complexity_name=complexity_name)
                    


def plot_all_titanic_tree_validation():
    vc = validation_curves()
    
    dh = data_helper()    
    X_train, X_test, y_train, y_test = dh.load_titanic_data_full_set()

    params_dict = {
            #'min_impurity_split': {'param_value': np.arange(0.0, 1.0, 0.02), 'reverse_xaxis': True},
            'max_depth': {'param_value': np.arange(1, 50, 1), 'reverse_xaxis': False} ,
            #'min_samples_split': {'param_value': np.arange(1, 200, 5), 'reverse_xaxis': True},
            #'min_samples_leaf': {'param_value': np.arange(2, 200, 5), 'reverse_xaxis': True},
            'max_leaf_nodes': {'param_value': np.arange(2, 200, 4), 'reverse_xaxis': False}#,
            #'max_features': {'param_value': np.arange(1, X_train_wine.shape[1], 1), 'reverse_xaxis': False}
            }
        
    
    outer_param_dict = { 'criterion': {'gini': params_dict}     
                         }
    
    vc.run(X_train, X_test, y_train, y_test, DecisionTreeClassifier, None, outer_param_dict, 'titanic', 'Tree, All Features', 'Tree nodes')

     


def plot_wine_boost_validation():
    vc = validation_curves()
    
    dh = data_helper()    
    X_train, X_test, y_train, y_test = dh.load_wine_data()

    params_dict = {
            #'min_impurity_split': {'param_value': np.arange(0.0, 1.0, 0.02), 'reverse_xaxis': True},
            'base_estimator__max_depth': {'param_value': np.arange(1, 50, 1), 'reverse_xaxis': False} ,
            #'min_samples_split': {'param_value': np.arange(1, 200, 5), 'reverse_xaxis': True},
            #'min_samples_leaf': {'param_value': np.arange(2, 200, 5), 'reverse_xaxis': True},
            'base_estimator__max_leaf_nodes': {'param_value': np.arange(2, 200, 4), 'reverse_xaxis': False}#,
            #'max_features': {'param_value': np.arange(1, X_train_wine.shape[1], 1), 'reverse_xaxis': False}
            }
        
    
    outer_param_dict = { 'algorithm': {'SAMME.R': params_dict}     
                         }
    
    vc.run(X_train, X_test, y_train, y_test, DecisionTreeClassifier, AdaBoostClassifier, outer_param_dict, 'wine', 'Boost, Feature Subset', '')


def plot_titanic_boost_validation():
    vc = validation_curves()
    
    dh = data_helper()    
    X_train, X_test, y_train, y_test = dh.load_titanic_data_full_set()

    params_dict = {
            #'min_impurity_split': {'param_value': np.arange(0.0, 1.0, 0.02), 'reverse_xaxis': True},
            'base_estimator__max_depth': {'param_value': np.arange(1, 50, 1), 'reverse_xaxis': False} ,
            #'min_samples_split': {'param_value': np.arange(1, 200, 5), 'reverse_xaxis': True},
            #'min_samples_leaf': {'param_value': np.arange(2, 200, 5), 'reverse_xaxis': True},
            'base_estimator__max_leaf_nodes': {'param_value': np.arange(2, 200, 4), 'reverse_xaxis': False}#,
            #'max_features': {'param_value': np.arange(1, X_train_wine.shape[1], 1), 'reverse_xaxis': False}
            }
        
    
    outer_param_dict = { 'algorithm': {'SAMME.R': params_dict}     
                         }
    
    vc.run(X_train, X_test, y_train, y_test, DecisionTreeClassifier, AdaBoostClassifier, outer_param_dict, 'Titanic', 'Boost', '')


def plot_all_wine_tree_validation():
    vc = validation_curves()
    
    dh = data_helper()    
    X_train_wine, X_test_wine, y_train_wine, y_test_wine =  dh.load_wine_data_full_set()

    params_dict = {
            #'min_impurity_split': {'param_value': np.arange(0.0, 1.0, 0.02), 'reverse_xaxis': True},
            'max_depth': {'param_value': np.arange(1, 50, 1), 'reverse_xaxis': False} ,
            #'min_samples_split': {'param_value': np.arange(1, 200, 5), 'reverse_xaxis': True},
            #'min_samples_leaf': {'param_value': np.arange(2, 200, 5), 'reverse_xaxis': True},
            'max_leaf_nodes': {'param_value': np.arange(2, 200, 4), 'reverse_xaxis': False}#,
            #'max_features': {'param_value': np.arange(1, X_train_wine.shape[1], 1), 'reverse_xaxis': False}
            }
        
    
    outer_param_dict = { 'criterion': {'gini': params_dict}     
                         }
    
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, DecisionTreeClassifier, None, outer_param_dict, 'wine', 'Tree, All Features', 'Tree nodes')


def plot_reduced_wine_tree_validation():
    vc = validation_curves()
    
    dh = data_helper()    
    X_train_wine, X_test_wine, y_train_wine, y_test_wine =  dh.load_wine_data()

    params_dict = {
            #'min_impurity_split': {'param_value': np.arange(0.0, 1.0, 0.02), 'reverse_xaxis': True},
            'max_depth': {'param_value': np.arange(1, 30, 1), 'reverse_xaxis': False} ,
            #'min_samples_split': {'param_value': np.arange(1, 200, 5), 'reverse_xaxis': True},
            #'min_samples_leaf': {'param_value': np.arange(2, 200, 5), 'reverse_xaxis': True},
            'max_leaf_nodes': {'param_value': np.arange(2, 200, 4), 'reverse_xaxis': False}#,
            #'max_features': {'param_value': np.arange(1, X_train_wine.shape[1], 1), 'reverse_xaxis': False}
            }
        
    
    outer_param_dict = { 'criterion': {'gini': params_dict}     
                         }
    
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, DecisionTreeClassifier, None, outer_param_dict, 'wine', 'Tree, Feature Subset', 'Tree nodes')


def plot_wine_knn_validation():
    vc = validation_curves()
    
    dh = data_helper()    
    X_train_wine, X_test_wine, y_train_wine, y_test_wine =  dh.load_wine_data_knn()

    params_dict = {'clf__n_neighbors': {'param_value': np.arange(1, 50, 1), 'reverse_xaxis': True}#,
                   #'clf__leaf_size': {'param_value': np.arange(1, 20, 1), 'reverse_xaxis': False},
                   #'clf__p': {'param_value': np.arange(1, 20, 1), 'reverse_xaxis': False}
                   }
    
    outer_param_dict = { 'clf__algorithm': {'kd_tree': params_dict}     
                         }
                     
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, KNeighborsClassifier, StandardScaler, outer_param_dict, 'wine', 'KNN', '')
        


def plot_titanic_knn_validation():
    vc = validation_curves()
    
    dh = data_helper()    
    X_train_wine, X_test_wine, y_train_wine, y_test_wine =  dh.load_titanic_data_full_set()

    params_dict = {'clf__n_neighbors': {'param_value': np.arange(1, 50, 1), 'reverse_xaxis': True}#,
                   #'clf__leaf_size': {'param_value': np.arange(1, 20, 1), 'reverse_xaxis': False},
                   #'clf__p': {'param_value': np.arange(1, 20, 1), 'reverse_xaxis': False}
                   }
    
    outer_param_dict = { 'clf__algorithm': {'kd_tree': params_dict}     
                         }
                     
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, KNeighborsClassifier, StandardScaler, outer_param_dict, 'Titanic', 'KNN', '')
        
          

def plot_wine_neural_validation():
    vc = validation_curves()
    
    dh = data_helper()    
    X_train_wine, X_test_wine, y_train_wine, y_test_wine =  dh.load_wine_data_knn()
    h = 100
                          
    params_dict = {
                    #'clf__max_iter': {'param_value': np.arange(1, 500, 10), 'reverse_xaxis': False},
                    #'clf__batch_size': {'param_value': np.arange(50,500,10), 'reverse_xaxis': False},
                    #'clf__learning_rate_init': {'param_value': np.arange(0.001,0.1,0.01), 'reverse_xaxis': False},
                    #'clf__power_t': {'param_value': np.arange(0.01,0.1,0.01), 'reverse_xaxis': False},
                    'clf__hidden_layer_sizes': {'param_value': [(h,),
                                                                (h,h,),
                                                                (h,h,h,),
                                                                (h,h,h,h,),
                                                                (h,h,h,h,h),
                                                                (h,h,h,h,h,h),
                                                                (h,h,h,h,h,h,h),
                                                                (h,h,h,h,h,h,h,h),
                                                                (h,h,h,h,h,h,h,h,h),
                                                                (h,h,h,h,h,h,h,h,h,h),
                                                                (h,h,h,h,h,h,h,h,h,h,h)
                                                                ], 'reverse_xaxis': False}
                    
    }
    
    outer_param_dict = { 'clf__activation': {#'identity': params_dict,
                                         #'logistic': params_dict,
                                         #'tanh': params_dict,
                                         'relu': params_dict}     
                         }
    
                         
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, MLPClassifier, StandardScaler, outer_param_dict, 'wine', 'Neural Net', '')
        




def plot_titanic_neural_validation2():
    vc = validation_curves()
    h = 100
    dh = data_helper()    
    X_train_wine, X_test_wine, y_train_wine, y_test_wine =  dh.load_titanic_data_full_set()

    params_dict = {
                    'clf__max_iter': {'param_value': np.arange(1, 10000, 1), 'reverse_xaxis': False},
                    #'clf__batch_size': {'param_value': np.arange(50,500,10), 'reverse_xaxis': False},
                    #'clf__learning_rate_init': {'param_value': np.arange(0.001,0.1,0.01), 'reverse_xaxis': False},
                    #'clf__power_t': {'param_value': np.arange(0.01,0.1,0.01), 'reverse_xaxis': False},
                    
    }
    
    outer_param_dict = { 'clf__activation': {#'identity': params_dict,
                                         #'logistic': params_dict,
                                         #'tanh': params_dict,
                                         'relu': params_dict}     
                         }
                         
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, MLPClassifier, StandardScaler, outer_param_dict, 'Titanic', 'Neural Net', '')
   
   
   
   

def plot_titanic_neural_validation():
    vc = validation_curves()
    h = 100
    dh = data_helper()    
    X_train_wine, X_test_wine, y_train_wine, y_test_wine =  dh.load_titanic_data_full_set()

    params_dict = {
                    #'clf__max_iter': {'param_value': np.arange(1, 500, 10), 'reverse_xaxis': False},
                    #'clf__batch_size': {'param_value': np.arange(50,500,10), 'reverse_xaxis': False},
                    #'clf__learning_rate_init': {'param_value': np.arange(0.001,0.1,0.01), 'reverse_xaxis': False},
                    #'clf__power_t': {'param_value': np.arange(0.01,0.1,0.01), 'reverse_xaxis': False},
                    'clf__hidden_layer_sizes': {'param_value': [(h,),
                                                                (h,h,),
                                                                (h,h,h,),
                                                                (h,h,h,h,),
                                                                (h,h,h,h,h),
                                                                (h,h,h,h,h,h),
                                                                (h,h,h,h,h,h,h),
                                                                (h,h,h,h,h,h,h,h),
                                                                (h,h,h,h,h,h,h,h,h),
                                                                (h,h,h,h,h,h,h,h,h,h),
                                                                (h,h,h,h,h,h,h,h,h,h,h)
                                                                ], 'reverse_xaxis': False}
                    
    }
    
    outer_param_dict = { 'clf__activation': {#'identity': params_dict,
                                         #'logistic': params_dict,
                                         #'tanh': params_dict,
                                         'relu': params_dict}     
                         }
                         
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, MLPClassifier, StandardScaler, outer_param_dict, 'Titanic', 'Neural Net', '')
   
   
   
   

def plot_wine_svc_validation():
    vc = validation_curves()
    
    dh = data_helper()    
    X_train_wine, X_test_wine, y_train_wine, y_test_wine =  dh.load_wine_data()

    params_dict = {#'clf__C': {'param_value': np.arange(1, 5, 1), 'reverse_xaxis': True},
                   #'clf__gamma': {'param_value': np.arange(0.1, 1., 0.05), 'reverse_xaxis': False},
                   'clf__degree': {'param_value': np.arange(1, 10, 1), 'reverse_xaxis': False}
                   }
    
    outer_param_dict = { 'clf__kernel': {'poly': params_dict}     
                         }
                     
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, SVC, StandardScaler, outer_param_dict, 'wine', 'SVM', '')
        


def plot_titanic_svc_validation():
    vc = validation_curves()
    
    dh = data_helper()    
    X_train_wine, X_test_wine, y_train_wine, y_test_wine =  dh.load_titanic_data_full_set()

    params_dict = {#'clf__C': {'param_value': np.arange(1, 5, 1), 'reverse_xaxis': True},
                   'clf__gamma': {'param_value': np.arange(0.1, 1., 0.05), 'reverse_xaxis': False},
                   #'clf__p': {'param_value': np.arange(1, 20, 1), 'reverse_xaxis': False}
                   }
    
    outer_param_dict = { 'clf__kernel': {'rbf': params_dict}     
                         }
                     
                     
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, SVC, StandardScaler, outer_param_dict, 'wine', 'SVM', '')
        
        

if __name__ == "__main__":
    
    #plot_wine_boost_validation()
    #plot_titanic_boost_validation()
    
    #plot_all_wine_tree_validation()
    #plot_all_titanic_tree_validation()
    
    #plot_reduced_wine_tree_validation()
    
    #plot_wine_knn_validation()
    #plot_titanic_knn_validation()
    
    
    #plot_wine_neural_validation()
    plot_titanic_neural_validation2()
    
    #plot_wine_svc_validation()
    #plot_titanic_svc_validation()
    
    '''
    vc = validation_curves()
    #vc.gridSearch2()
    
    dh = data_helper()    
    X_train_titanic, X_test_titanic, y_train_titanic, y_test_titanic =  dh.load_titanic_data()
    X_train_wine, X_test_wine, y_train_wine, y_test_wine =  dh.load_wine_data_full_set()



    ###
    ### SVM
    ###
    
    params_dict_rbf = {'clf__C': {'param_value': np.arange(1, 500, 10), 'reverse_xaxis': False},
                       'clf__max_iter': {'param_value': np.arange(0, 300, 1), 'reverse_xaxis': True},
                       'clf__gamma': {'param_value': np.arange(1, 30, 1), 'reverse_xaxis': False},
                       'clf__tol': {'param_value': np.arange(0.000001, 2.0, 0.005), 'reverse_xaxis': True}
                       }
    
    params_dict_poly = {'clf__degree': {'param_value': np.arange(0, 15, 1), 'reverse_xaxis': True}
                        }
    
    params_dict_linear = {'clf__max_iter': {'param_value': np.arange(0, 300, 1), 'reverse_xaxis': True},
                          'clf__tol': {'param_value': np.arange(0.000001, 2.0, 0.005), 'reverse_xaxis': True}
                          }
    
    outer_param_dict = { 'clf__kernel': {'rbf': params_dict_rbf,
                                         'poly': params_dict_poly,
                                         'linear': params_dict_linear,
                                         'sigmoid': params_dict_poly}     
                         }

    vc.run(X_train_titanic, X_test_titanic, y_train_titanic, y_test_titanic, SVC, StandardScaler, outer_param_dict, 'titanic', 'SVM', '')
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, SVC, StandardScaler,  outer_param_dict, 'wine', 'SVM', '')
        
        
    ###
    ### KNN
    ###
    params_dict = {'clf__n_neighbors': {'param_value': np.arange(1, 500, 10), 'reverse_xaxis': True},
                   'clf__leaf_size': {'param_value': np.arange(1, 20, 1), 'reverse_xaxis': False},
                   'clf__p': {'param_value': np.arange(1, 20, 1), 'reverse_xaxis': False}
                   }
    
    outer_param_dict = { 'clf__algorithm': {'auto': params_dict,
                                         'ball_tree': params_dict,
                                         'kd_tree': params_dict,
                                         'brute': params_dict}     
                         }
                     
    vc.run(X_train_titanic, X_test_titanic, y_train_titanic, y_test_titanic, KNeighborsClassifier, StandardScaler, outer_param_dict, 'titanic', 'KNN', '')
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, KNeighborsClassifier, StandardScaler, outer_param_dict, 'wine', 'KNN', '')
        

    ###
    ### TREE
    ###
    
    # max features should be 1 to n_features
    params_dict = {
            'min_impurity_split': {'param_value': np.arange(0.0, 1.0, 0.02), 'reverse_xaxis': True},
            'max_depth': {'param_value': np.arange(1, 25, 1), 'reverse_xaxis': False},
            'min_samples_split': {'param_value': np.arange(1, 200, 5), 'reverse_xaxis': True},
            'min_samples_leaf': {'param_value': np.arange(2, 200, 5), 'reverse_xaxis': True},
            'max_leaf_nodes': {'param_value': np.arange(2, 225, 5), 'reverse_xaxis': False},
            'max_features': {'param_value': np.arange(1, X_train_titanic.shape[1], 1), 'reverse_xaxis': False}
            }
        
    
    outer_param_dict = { 'criterion': {'gini': params_dict,
                                         'entropy': params_dict}     
                         }
    
    vc.run(X_train_titanic, X_test_titanic, y_train_titanic, y_test_titanic, DecisionTreeClassifier, None, outer_param_dict, 'titanic', 'Tree', 'Tree nodes')
    
    
    params_dict = {
            'min_impurity_split': {'param_value': np.arange(0.0, 1.0, 0.02), 'reverse_xaxis': True},
            'max_depth': {'param_value': np.arange(1, 25, 1), 'reverse_xaxis': False},
            'min_samples_split': {'param_value': np.arange(1, 200, 5), 'reverse_xaxis': True},
            'min_samples_leaf': {'param_value': np.arange(2, 200, 5), 'reverse_xaxis': True},
            'max_leaf_nodes': {'param_value': np.arange(2, 225, 5), 'reverse_xaxis': False},
            'max_features': {'param_value': np.arange(1, X_train_wine.shape[1], 1), 'reverse_xaxis': False}
            }
        
    
    outer_param_dict = { 'criterion': {'gini': params_dict,
                                         'entropy': params_dict}     
                         }
    
    
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, DecisionTreeClassifier, None, outer_param_dict, 'wine', 'Tree', 'Tree nodes')


    ###
    ### Neural
    ###
    params_dict = {
                    'clf__max_iter': {'param_value': np.arange(1, 500, 10), 'reverse_xaxis': False},
                    'clf__batch_size': {'param_value': np.arange(50,500,10), 'reverse_xaxis': False},
                    'clf__learning_rate_init': {'param_value': np.arange(0.001,0.1,0.01), 'reverse_xaxis': False},
                    'clf__power_t': {'param_value': np.arange(0.01,0.1,0.01), 'reverse_xaxis': False}
    }
    
    outer_param_dict = { 'clf__activation': {'identity': params_dict,
                                         'logistic': params_dict,
                                         'tanh': params_dict,
                                         'relu': params_dict}     
                         }
    
    
    vc.run(X_train_titanic, X_test_titanic, y_train_titanic, y_test_titanic, MLPClassifier, StandardScaler, outer_param_dict, 'titanic', 'Neural Net', '')
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, MLPClassifier, StandardScaler, outer_param_dict, 'wine', 'Neural Net', '')
        
   
    
    ###
    ### Boosting
    ###
    tree = DecisionTreeClassifier(criterion=criterion)
    estimator = AdaBoostClassifier(base_estimator=tree, random_state=0, n_estimators=260, learning_rate=learning_rate)
    
    params_dict = {
                    'clf__max_iter': {'param_value': np.arange(1, 500, 10), 'reverse_xaxis': False},
                    'clf__batch_size': {'param_value': np.arange(50,500,10), 'reverse_xaxis': False},
                    'clf__learning_rate_init': {'param_value': np.arange(0.001,0.1,0.01), 'reverse_xaxis': False},
                    'clf__power_t': {'param_value': np.arange(0.01,0.1,0.01), 'reverse_xaxis': False}
    }
    
    outer_param_dict = { 'clf__activation': {'identity': params_dict,
                                         'logistic': params_dict,
                                         'tanh': params_dict,
                                         'relu': params_dict}     
                         }
    
    
    vc.run(X_train_titanic, X_test_titanic, y_train_titanic, y_test_titanic, MLPClassifier, StandardScaler, outer_param_dict, 'titanic', 'Neural Net', '')
    vc.run(X_train_wine, X_test_wine, y_train_wine, y_test_wine, MLPClassifier, StandardScaler, outer_param_dict, 'wine', 'Neural Net', '')
    '''
    

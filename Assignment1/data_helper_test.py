from data_helper import *
import matplotlib.pyplot as plt
import uuid

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, Imputer
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import learning_curve, validation_curve



from timeit import default_timer as timer

def run_learning_curves(X_train, X_test, y_train, y_test, pipeline, title_desc=''):
        
    
    # plot the learning curves using sklearn and matplotlib
    plt.clf()
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipeline,
                                                            X=X_train,
                                                            y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_scores = 1. - train_scores
    test_scores = 1. - test_scores
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5,
             label='training error')
    
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean,
             color='green', marker='s',
             markersize=5, linestyle='--',
             label='validation error')        
    
    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    
    plt.grid()
    plt.title("Learning curve: %s" % (title_desc))
    plt.xlabel('Number of training samples')
    plt.ylabel('Error')
    plt.legend(loc='lower right')
    fn = './output/' + str(uuid.uuid4()) + '_learningcurve.png'
    plt.savefig(fn)


def run_validation_curves(X_train, X_test, y_train, y_test, pipeline, param_name, param_range, title_desc=''):
    
    #X_train, X_test, y_train, y_test =  dh.load_titanic_data()
    #X_train, X_test, y_train, y_test =  dh.load_nba_data()
    
    # plot the validation curves
    plt.clf()
    
    train_scores, test_scores = validation_curve(estimator=pipeline,
                                                 X=X_train,
                                                 y=y_train, 
                                                 param_name=param_name, 
                                                 param_range=param_range)
    
    train_scores = 1. - train_scores
    test_scores = 1. - test_scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5,
             label='training error')
    
    plt.fill_between(param_range,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')
    
    plt.plot(param_range, test_mean,
             color='green', marker='s',
             markersize=5, linestyle='--',
             label='validation error')
    
    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    
    plt.grid()
    plt.title("Validation curve: %s" % (title_desc))
    plt.xlabel(param_name)
    plt.ylabel('Error')
    plt.legend(loc='lower right')
    fn = './output/' + str(uuid.uuid4()) + '_validationcurve.png'
    plt.savefig(fn)
    

def run_grid_search(pipeline, parameters):
    dh = data_helper()
    #X_train, X_test, y_train, y_test =  dh.load_wine_data()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data()
    
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
        
def grid_search_svm():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', SVC(random_state=0, max_iter=500, tol=0.001))])
    
    
    parameters = {'clf__kernel': ('rbf', 'poly', 'linear'),
                      'clf__C': np.arange(1., 3., 1.),
                      #'clf__max_iter': np.arange(200., 800., 25.),
                      'clf__gamma': np.arange(0., 10., 0.5),
                      #'clf__tol': np.arange(0.000001, 2.0, 0.5)
                      }
    run_grid_search(pipeline, parameters)
    
    
def grid_search_tree():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', DecisionTreeClassifier(random_state=0))])
    
    parameters = {'clf__criterion': ('gini', 'entropy'),
                  #'clf__min_impurity_split': np.arange(0, 0.5, 0.01),
                  'clf__max_depth': np.arange(1, 40, 1)#,
                  #'clf__min_samples_split': np.arange(1, 200, 5),
                  #'clf__min_samples_leaf': np.arange(2, 200, 5),
                  #'clf__min_weight_fraction_leaf': np.arange(0.0, 0.5, 0.05),
                  #'clf__max_leaf_nodes': np.arange(2, 300, 5)
                  }
    
    run_grid_search(pipeline, parameters)

def grid_search_neural():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', MLPClassifier(random_state=0))])
    
    parameters = {'clf__activation': ('identity', 'logistic', 'tanh', 'relu'),
                  'clf__solver': ('lbgfs', 'sgd', 'adam'),
                  'clf__learning_rate': ('constant', 'invscaling', 'adaptive'),
                  'clf__shuffle': (True, False),
                  'clf__learning_rate_init': (1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001)
                  #'clf__max_iter': np.arange(1, 500, 10),
                  #'clf__batch_size': np.arange(50,500,10),
                  #'clf__learning_rate_init': np.arange(0.001,0.1,0.01),
                  #'clf__power_t': np.arange(0.01,0.1,0.01)
                  }
    
    run_grid_search(pipeline, parameters)


def grid_search_neural_sgd():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', MLPClassifier(solver='sgd', random_state=0))])
    
    parameters = {'clf__alpha': (0.01, 0.001),
                  'clf__activation': ('identity', 'logistic', 'tanh', 'relu'),
                  'clf__learning_rate': ('constant', 'invscaling', 'adaptive'),
                  'clf__shuffle': (True, False),
                  'clf__learning_rate_init': (0.01, 0.001, 0.0001),
                  'clf__power_t': (0.0, 0.5, 1.0),
                  'clf__momentum': np.arange(0.0, 1.0, 0.1),
                  'clf__nesterovs_momentum': (True, False)
                  #'clf__max_iter': np.arange(1, 500, 10),
                  #'clf__batch_size': np.arange(50,500,10),
                  #'clf__learning_rate_init': np.arange(0.001,0.1,0.01),
                  #'clf__power_t': np.arange(0.01,0.1,0.01)
                  }
    
    run_grid_search(pipeline, parameters)



def run_learning_curves_bag():
    
    pipeline = Pipeline([('clf', BaggingClassifier(DecisionTreeClassifier(max_leaf_nodes=38,random_state=0)))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Bagging')


def run_learning_curves_boost():
    
    pipeline = Pipeline([('clf', AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=38,random_state=0)))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Boosting')
            

def run_learning_curves_tree_full_data():
    pipeline = Pipeline([('clf', DecisionTreeClassifier(max_leaf_nodes=38,random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_full_set()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Tree (all data)')
    
    
    
def run_validation_curves_tree_full_data_max_depth():
    pipeline = Pipeline([('clf', DecisionTreeClassifier(random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_full_set()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__max_depth', np.arange(1,50,1),
                          'Decision Tree For All Wine Data')
    

def run_validation_curves_boost():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', AdaBoostClassifier(DecisionTreeClassifier(random_state=0), 100))])
    #AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=38,random_state=0))
    parameters = {'clf__criterion': ('gini', 'entropy'),
                  #'clf__min_impurity_split': np.arange(0, 0.5, 0.01),
                  'clf__max_depth': np.arange(1, 40, 1)#,
                  #'clf__min_samples_split': np.arange(1, 200, 5),
                  #'clf__min_samples_leaf': np.arange(2, 200, 5),
                  #'clf__min_weight_fraction_leaf': np.arange(0.0, 0.5, 0.05),
                  #'clf__max_leaf_nodes': np.arange(2, 300, 5)
                  }
    
    param_name = ['max_depth',
                  'min_impurity_split',
                  'max_leaf_nodes']
    
    param_range = [np.arange(1,30,1),
                   np.arange(0, 0.25, 0.005),
                   np.arange(1, 50, 1)]
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data()

    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__base_estimator__max_leaf_nodes', np.arange(2,50,1),
                          'Boost')
    
    
if __name__ == "__main__":
    #run_learning_curves_bag()
    #run_validation_curves_tree_full_data_max_depth()
    run_learning_curves_tree_full_data()    
    '''
    grid_search_neural_sgd()
    
    
    
    dh = data_helper()
    
    
    X_train, X_test, y_train, y_test = dh.load_wine_data()
    
    
    skf = StratifiedKFold(n_splits=5, random_state=0)
    
    out_sample_errors = []
    in_sample_errors = []
    complexity_measures = []
    fit_times = []
    predict_times = []
    
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    
    # do the cross validation
    for k, (train, test) in enumerate(skf.split(X=X_train, y=y_train)):
        
        #start = timer()
        # run the learning algorithm
        clf.fit(X_train[train], y_train[train])
        #end = timer()
        #tot_fit_time = (end - start) * 1000.
        #fit_times.append(tot_fit_time)
        
        # in sample
        #start = timer()
        predicted_values = clf.predict(X_train[train])
        #end = timer()
        #tot_in_sample_predict_time = (end - start) * 1000.
        #predict_times.append(tot_in_sample_predict_time)
        
        in_sample_mse = mean_squared_error(y_train[train], predicted_values)
        in_sample_errors.append(in_sample_mse)
        
        # out of sample
        #start = timer()
        predicted_values = clf.predict(X_train[test])
        #end = timer()
        #tot_out_sample_predict_time = (end - start) * 1000.
        #predict_times.append(tot_out_sample_predict_time)
        out_sample_mse = mean_squared_error(y_train[test], predicted_values)
        out_sample_errors.append(out_sample_mse)

        print('Fold:', k+1, ', Validation error: ', out_sample_mse,
              ', Training error: ', in_sample_mse)
       
       
    '''
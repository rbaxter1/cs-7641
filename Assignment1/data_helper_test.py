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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, Imputer

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


def run_validation_curves(X_train, X_test, y_train, y_test, pipeline, param_name, param_range, title_desc='', param_range_plot=None):
    
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
    
    if param_range_plot != None:
        param_range = param_range_plot
        
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

def run_grid_search_data(X_train, X_test, y_train, y_test, pipeline, parameters):    
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

def grid_search_wine_neural():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', MLPClassifier(random_state=0,
                                               max_iter=500
                                               ))])
    
    parameters = {'clf__activation': ('identity', 'logistic', 'tanh', 'relu'),
                  'clf__solver': ('lbgfs', 'sgd', 'adam'),
                  'clf__learning_rate': ('constant', 'invscaling', 'adaptive'),
                  'clf__shuffle': (True, False),
                  #'clf__learning_rate_init': np.arange(0.001,0.01,0.001)
                  #'clf__max_iter': np.arange(1, 500, 10),
                  #'clf__batch_size': np.arange(50,500,10),
                  #'clf__learning_rate_init': np.arange(0.001,0.1,0.01),
                  #'clf__power_t': np.arange(0.01,0.1,0.01)
                  #'clf__hidden_layer_sizes', [(10),(10,10),(10,10,10),(10,10,10,10),(10,10,10,10,10)]
                  }
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_knn()


    run_grid_search_data(X_train, X_test, y_train, y_test, pipeline, parameters)


def grid_search_wine_neural2():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', MLPClassifier(activation='relu',
                                               learning_rate='constant',
                                               shuffle=True,
                                               solver='adam',
                                               random_state=0,
                                               max_iter=500,
                                               batch_size=60,
                                               ))])
    
    parameters = {#'clf__activation': ('identity', 'logistic', 'tanh', 'relu'),
                  #'clf__solver': ('lbgfs', 'sgd', 'adam'),
                  #'clf__learning_rate': ('constant', 'invscaling', 'adaptive'),
                  #'clf__shuffle': (True, False),
                  #'clf__learning_rate_init': np.arange(0.001,0.01,0.001)
                  #'clf__max_iter': np.arange(1, 500, 10),
                  #'clf__batch_size': np.arange(50,500,10),
                  #'clf__learning_rate_init': np.arange(0.001,0.1,0.01),
                  #'clf__power_t': np.arange(0.01,0.1,0.01)
                  #'clf__hidden_layer_sizes', [(10),(10,10),(10,10,10),(10,10,10,10),(10,10,10,10,10)]
                  }
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_knn()


    run_grid_search_data(X_train, X_test, y_train, y_test, pipeline, parameters)



def grid_search_titanic_neural():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', MLPClassifier(random_state=0,
                                               max_iter=500
                                               ))])
    
    parameters = {'clf__activation': ('identity', 'logistic', 'tanh', 'relu'),
                  'clf__solver': ('lbgfs', 'sgd', 'adam'),
                  'clf__learning_rate': ('constant', 'invscaling', 'adaptive'),
                  'clf__shuffle': (True, False),
                  #'clf__learning_rate_init': np.arange(0.001,0.01,0.001)
                  #'clf__max_iter': np.arange(1, 500, 10),
                  #'clf__batch_size': np.arange(50,500,10),
                  #'clf__learning_rate_init': np.arange(0.001,0.1,0.01),
                  #'clf__power_t': np.arange(0.01,0.1,0.01)
                  #'clf__hidden_layer_sizes', [(10),(10,10),(10,10,10),(10,10,10,10),(10,10,10,10,10)]
                  }
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data_full_set()


    run_grid_search_data(X_train, X_test, y_train, y_test, pipeline, parameters)





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
            

def run_learning_curves_tree_wine_full_data_max_depth(max_depth):
    pipeline = Pipeline([('clf', DecisionTreeClassifier(max_depth=max_depth,random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_full_set()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Tree, Wine, All Features, max_depth=' + str(max_depth))
    
    

def run_learning_curves_tree_wine_max_depth(max_depth):
    pipeline = Pipeline([('clf', DecisionTreeClassifier(max_depth=max_depth,random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Tree, Wine, Feature Subset, max_depth=' + str(max_depth))
    
    
def run_learning_curves_knn_wine_k(k):
    pipeline = Pipeline([ ('scl', StandardScaler() ),
                          ('clf', KNeighborsClassifier())])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_knn()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'KNN, Wine, K=' + str(k))
    
def run_learning_curves_knn_titanic_k(k):
    pipeline = Pipeline([ ('scl', StandardScaler() ),
                          ('clf', KNeighborsClassifier(algorithm='kd_tree'))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data_full_set()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'KNN, Titantic, K=' + str(k))

    


def run_learning_curves_tree_titanic_data_max_depth(max_depth):
    pipeline = Pipeline([('clf', DecisionTreeClassifier(max_depth=max_depth,random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Tree, Titanic, All Features, max_depth=' + str(max_depth))


def run_learning_curves_tree_titanic_full_data_max_depth(max_depth):
    pipeline = Pipeline([('clf', DecisionTreeClassifier(max_depth=max_depth,random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data_full_set()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Tree, Titanic, All Features, max_depth=' + str(max_depth))
    

def run_learning_curves_neural_wine_data():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', MLPClassifier(activation='relu',
                                               learning_rate='constant',
                                               shuffle=True,
                                               solver='adam',
                                               random_state=0,
                                               max_iter=500,
                                               batch_size=60,
                                               ))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_knn()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Neural, Wine, Feature subset')




def run_learning_curves_boost_wine_max_depth(max_depth):
    pipeline = Pipeline([('clf', AdaBoostClassifier(DecisionTreeClassifier(random_state=0)))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Boost, Wine, Feature Subset, max_depth=' + str(max_depth))


def run_learning_curves_boost_titanic_full_data_max_depth(max_depth):
    pipeline = Pipeline([('clf', AdaBoostClassifier(DecisionTreeClassifier(random_state=0)))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data_full_set()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Boost, Titanic, All Features, max_depth=' + str(max_depth))


def run_learning_curves_neural_wine():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', MLPClassifier(activation='relu',
                                               learning_rate='constant',
                                               shuffle=True,
                                               solver='adam',
                                               random_state=0,
                                               max_iter=500,
                                               batch_size=60,
                                               hidden_layer_sizes=(100,)
                                               ))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_knn()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Neural, Wine, Feature Subset, hidden_layers=1')


def run_learning_curves_neural_titanic():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', MLPClassifier(random_state=0,
                                               max_iter=50,
                                               activation='relu',
                                               shuffle=True,
                                               solver='adam',
                                               learning_rate_init=0.001,
                                               learning_rate='constant',
                                               hidden_layer_sizes=(100,)
                                               ))])
        
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data_full_set()

    run_learning_curves(X_train, X_test, y_train, y_test, pipeline, 'Neural, Titanic, Feature Subset, hidden_layers=1')





 
    
    
    

    
def run_validation_curves_tree_wine_max_depth():
    pipeline = Pipeline([('clf', DecisionTreeClassifier(random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__max_depth', np.arange(1,50,1),
                          'Tree, Wine, Features Subset')

def run_validation_curves_tree_wine_max_leaf_nodes():
    pipeline = Pipeline([('clf', DecisionTreeClassifier(random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__max_leaf_nodes', np.arange(2,200,4),
                          'Tree, Wine, Features Subset')
    
    
def run_validation_curves_tree_wine_full_data_max_depth():
    pipeline = Pipeline([('clf', DecisionTreeClassifier(random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_full_set()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__max_depth', np.arange(1,50,1),
                          'Tree, Wine, All Features')

def run_validation_curves_tree_wine_full_data_max_leaf_nodes():
    pipeline = Pipeline([('clf', DecisionTreeClassifier(random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_full_set()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__max_leaf_nodes', np.arange(2,200,4),
                          'Tree, Wine, All Features')
    

def run_validation_curves_tree_titanic_full_data_max_depth():
    pipeline = Pipeline([('clf', DecisionTreeClassifier(random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data_full_set()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__max_depth', np.arange(1,50,1),
                          'Tree, Titanic, All Features')



def run_validation_curves_tree_titanic_full_data_max_leaf_nodes():
    pipeline = Pipeline([('clf', DecisionTreeClassifier(random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data_full_set()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__max_leaf_nodes', np.arange(2,200,4),
                          'Tree, Titanic, All Features')

def run_validation_curves_tree_titanic_data_max_leaf_nodes():
    pipeline = Pipeline([('clf', DecisionTreeClassifier(random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__max_leaf_nodes', np.arange(2,200,4),
                          'Tree, Titanic')


def run_validation_curves_tree_titanic_data_max_depth():
    pipeline = Pipeline([('clf', DecisionTreeClassifier(random_state=0))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__max_depth', np.arange(1,50,1),
                          'Tree, Titanic')
    
    

def run_validation_curves_knn_titanic_data_k():
    pipeline = Pipeline([ ('scl', StandardScaler() ),
                          ('clf', KNeighborsClassifier(algorithm='kd_tree'))])

    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data_full_set()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__n_neighbors', np.arange(1,30,1),
                          'KNN, Titanic')
    


def run_validation_curves_knn_wine_data_k():
    pipeline = Pipeline([ ('scl', StandardScaler() ),
                          ('clf', KNeighborsClassifier(algorithm='kd_tree'))])

    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_knn()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__n_neighbors', np.arange(1,50,1),
                          'KNN, Wine Feature Subset')
    


def run_validation_curves_neural_titanic_data_hidden_layer_sizes():
    pipeline = Pipeline([ ('scl', StandardScaler() ),
                          ('clf', MLPClassifier())])

    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data_full_set()
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__hidden_layer_sizes', [(1,),(1,1,),(1,1,1),(1,1,1,1),(1,1,1,1,1)],
                          'Neural, Titanic')
    
def run_validation_curves_neural_titanic_data_hidden_layer_sizes():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', MLPClassifier(random_state=0,
                                               max_iter=50,
                                               activation='relu',
                                               shuffle=True,
                                               solver='adam',
                                               learning_rate_init=0.001,
                                               learning_rate='constant'
                                               ))])
        
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_titanic_data_full_set()

    h = 100
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__hidden_layer_sizes', [(h,),
                                                      (h,h,),
                                                      (h,h,h,),
                                                      (h,h,h,h),
                                                      (h,h,h,h,h),
                                                      (h,h,h,h,h,h),
                                                      (h,h,h,h,h,h,h),
                                                      (h,h,h,h,h,h,h,h),
                                                      (h,h,h,h,h,h,h,h,h),
                                                      (h,h,h,h,h,h,h,h,h,h),
                                                      (h,h,h,h,h,h,h,h,h,h,h)
                                                      ],
                          'Neural, Wine Feature Subset', [1,
                                                          2,
                                                          3,
                                                          4,
                                                          5,
                                                          6,
                                                          7,
                                                          8,
                                                          9,
                                                          10,
                                                          11
                                                          ])
    


def run_validation_curves_neural_wine_data_hidden_layer_sizes():
    pipeline = Pipeline([('scl', StandardScaler()),
                         ('clf', MLPClassifier(activation='relu',
                                               learning_rate='constant',
                                               shuffle=True,
                                               solver='adam',
                                               random_state=0,
                                               max_iter=500,
                                               batch_size=60,
                                               ))])
    
    dh = data_helper()
    X_train, X_test, y_train, y_test =  dh.load_wine_data_knn()

    
    h = 100
    
    run_validation_curves(X_train, X_test, y_train, y_test, pipeline,
                          'clf__hidden_layer_sizes', [(h,),
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
                                                      ],
                          'Neural, Wine Feature Subset', [1,
                                                          2,
                                                          3,
                                                          4,
                                                          5,
                                                          6,
                                                          7,
                                                          8,
                                                          9,
                                                          10,
                                                          11
                                                          ])
    
    
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
    '''
    run_validation_curves_tree_wine_full_data_max_depth()
    run_validation_curves_tree_wine_full_data_max_leaf_nodes()
    run_learning_curves_tree_wine_full_data_max_depth(8)

    run_validation_curves_tree_titanic_full_data_max_depth()
    run_validation_curves_tree_titanic_full_data_max_leaf_nodes()
    run_learning_curves_tree_titanic_full_data_max_depth(6)
    
    run_validation_curves_tree_wine_max_depth()
    run_validation_curves_tree_wine_max_leaf_nodes()
    run_learning_curves_tree_wine_max_depth(4)
    
    run_validation_curves_knn_wine_data_k()
    run_learning_curves_knn_wine_k(35)
    
    run_validation_curves_knn_titanic_data_k()
    run_learning_curves_knn_titanic_k(3)
    '''
    #grid_search_wine_neural()
    '''
    Best score: 0.742
    Best parameters set:
	clf__activation: 'relu'
	clf__learning_rate: 'constant'
	clf__shuffle: True
	clf__solver: 'adam'
    '''
    
    #grid_search_wine_neural2()
    '''    
    Best score: 0.735
    Best parameters set:
	clf__activation: 'relu'
	clf__learning_rate: 'constant'
	clf__shuffle: True
	clf__solver: 'adam'
    '''
    
    #grid_search_titanic_neural()
    '''
    Best score: 0.809
    Best parameters set:
	clf__activation: 'relu'
	clf__learning_rate: 'constant'
	clf__shuffle: True
	clf__solver: 'adam'
    '''
    
    run_learning_curves_boost_wine_max_depth(1)    
    run_learning_curves_boost_titanic_full_data_max_depth(1)
    
    
    
    #run_validation_curves_neural_wine_data_hidden_layer_sizes()
    #run_learning_curves_neural_wine_data()
    
    #run_validation_curves_neural_titanic_data_hidden_layer_sizes()
    #run_validation_curves_neural_titanic_data_hidden_layer_sizes()




    #run_validation_curves_tree_titanic_data_max_depth()
    #run_validation_curves_tree_titanic_data_max_leaf_nodes()
    #run_learning_curves_tree_titanic_data_max_depth(8)


    
    '''
    #run_learning_curves_bag()
        
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
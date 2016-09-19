from data_helper import *
import matplotlib.pyplot as plt


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
from sklearn.ensemble import AdaBoostClassifier

if __name__ == "__main__":
    dh = data_helper()
    
    
    X_train, X_test, y_train, y_test = dh.load_wine_data_orig()
    
    
    skf = StratifiedKFold(n_splits=5, random_state=0)
    
    out_sample_errors = []
    in_sample_errors = []
    complexity_measures = []
    fit_times = []
    predict_times = []
    
    clf = DecisionTreeClassifier()
    
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
       
       
       
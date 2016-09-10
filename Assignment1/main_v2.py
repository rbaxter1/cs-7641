import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split

from tree_test import *
from knn_test import *
from svm_test import *
from boost_test import *
from neural_test import *

from timeit import default_timer as timer

import io
import pydotplus

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    df = pd.read_csv('./data/shot_logs.csv', sep=',')
    
    #x_col_names = ['player_id', 'SHOT_DIST', 'TOUCH_TIME']
    #df = pd.get_dummies(df[['SHOT_RESULT', 'LOCATION', 'Age', 'Survived']])
    
    
    le = LabelEncoder()
    le.fit(df['LOCATION'])
    le.transform(df['LOCATION']) 
    df['LOCATION_ENC'] = le.transform(df['LOCATION'])
        
    le = LabelEncoder()
    le.fit(df['SHOT_RESULT'])
    le.transform(df['SHOT_RESULT']) 
    df['SHOT_RESULT_ENC'] = le.transform(df['SHOT_RESULT'])
        
    x_col_names = ['player_id', 'SHOT_DIST', 'TOUCH_TIME', 'LOCATION_ENC', 'PTS_TYPE', 'DRIBBLES', 'SHOT_NUMBER', 'FINAL_MARGIN']
    x, y = df.loc[:,x_col_names].values, df.loc[:,'SHOT_RESULT_ENC'].values
    
    # split the data into training and test data
    # for the wine data using 30% of the data for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    
    
    
    myModel = rb_knn_test(x_train, x_test, y_train, y_test, x_col_names, 'nba_knn', cv=5)
    '''
    start = timer()
    myModel.run_model(n_neighbors=20, leaf_size=30, p=5, do_plot=False)
    end = timer()
    print('redwine_knn run_model took:', end - start)
    '''
    
    start = timer()
    train, test = myModel.run_cv_model(n_neighbors=20, leaf_size=30, p=5, do_plot=False)
    end = timer()
    t = end - start
    print('nba_knn run_cv_model took:', t)
    ''
    perf.loc[len(perf)] = ['Wine', 'Knn', train, test, t]
    
    start = timer()
    myModel.plot_validation_curve(n_neighbors=20, leaf_size=30, p=5)
    end = timer()
    print('redwine_knn plot_validation_curve took:', end - start)    
    '''
    
    clf = DecisionTreeClassifier(criterion='gini', max_depth=5, min_impurity_split=0.25)
    #, min_impurity_split=0.1
    #params = clf.get_params()
    #params['criterion'] = 'entropy'
    #params['max_depth'] = 3
    #clf.set_params(**params)
    
    clf.fit(x_train, y_train)
    '''
    dot_data = io.StringIO()
    export_graphviz(clf,
                    out_file=dot_data,
                    feature_names=x_col_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    fn = './output/no_prune_graph.pdf'
    graph.write_pdf(fn)
    '''
    
    #params['min_impurity_split'] = 3
    #clf.set_params(**params)
    clf = DecisionTreeClassifier(criterion='entropy', min_impurity_split=3)
    clf.fit(x_train, y_train)
    
    
    myModel = rb_tree_test(x_train, x_test, y_train, y_test, x_col_names, 'nba_tree', cv=5)
    
    myModel.plot_validation_curve2(max_depth=4, criterion='entropy')
    
    
    '''
    dot_data = io.StringIO()
    export_graphviz(clf,
                    out_file=dot_data,
                    feature_names=x_col_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    fn = './output/prune_graph.pdf'
    graph.write_pdf(fn)
    '''
    
    '''
    myModel = rb_tree_test(x_train, x_test, y_train, y_test, x_col_names, 'nba_tree', cv=5)
    
    start = timer()
    myModel.run_model(max_depth=4, criterion='entropy', do_plot=False)
    end = timer()
    print('nba_tree run_model took:', end - start)
    
    start = timer()
    train, test = myModel.run_cv_model(max_depth=4, criterion='entropy', do_plot=False)
    end = timer()
    t = end - start
    print('nba_tree run_cv_model took:', t)
    
    #perf.loc[len(perf)] = ['Wine', 'Tree', train, test, t]
    
    start = timer()
    myModel.plot_validation_curve(max_depth=4, criterion='entropy')
    end = timer()
    print('redwine_tree plot_validation_curve took:', end - start)
    
    '''
    
    #df = pd.read_csv('./data/face_training.csv', sep=',')
    
    # Use Pythagorean theorem to calculate segment distance
    '''
    df['left_eye_outer_corner_x']
    df['left_eye_outer_corner_y']
    df['right_eye_outer_corner_x']
    df['right_eye_outer_corner_y']
    df['nose_tip_x']
    df['nose_tip_y']
    '''
    
    
    #(df['left_eye_outer_corner_y'] - df['right_eye_outer_corner_y']).pow(2)
    
    #data.size - np.isnan(data).sum()
    
    #data = df.values
    
    #print(df[0])
    
    # PREPARE DATA
    # this data set is missing some values. where missing, we could
    # impute a value like average or median. or remove the rows having
    # missing data. the disadvantage of removing values is we may be taking
    # away valuable information that the learning algorithm could use.
    
    # calculate segment length using Pythagorean theorem
    # distance forumla = delta x^2 + delta y^2 = length^2
    
    # eyes
    #df['right_eye_len'] = ((df['right_eye_outer_corner_x'] - df['right_eye_inner_corner_x']).pow(2) + (df['right_eye_outer_corner_y'] - df['right_eye_inner_corner_y']).pow(2)).pow(0.5)
    #df['left_eye_len'] = ((df['left_eye_outer_corner_x'] - df['left_eye_inner_corner_x']).pow(2) + (df['left_eye_outer_corner_y'] - df['left_eye_inner_corner_y']).pow(2)).pow(0.5)
    #df['outer_eye_span'] = ((df['left_eye_outer_corner_x'] - df['right_eye_outer_corner_x']).pow(2) + (df['left_eye_outer_corner_y'] - df['right_eye_outer_corner_y']).pow(2)).pow(0.5)
    #df['inner_eye_span'] = ((df['left_eye_inner_corner_x'] - df['right_eye_inner_corner_x']).pow(2) + (df['left_eye_inner_corner_y'] - df['right_eye_inner_corner_y']).pow(2)).pow(0.5)
    
    #imr = Imputer(strategy='mean')
    #imr.fit(df['right_eye_len'].reshape(-1, 1))
    #imputed_data = imr.transform(df['right_eye_len'].reshape(-1, 1))
    
    #df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
    #df.plot.scatter(x='a', y='b');
    
    
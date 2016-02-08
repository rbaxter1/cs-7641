import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.cross_validation import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.pipeline import Pipeline

from mlxtend.evaluate import plot_decision_regions, plot_learning_curves

import io
import pydotplus


class rb_tree:
    
    def __init__(self, x, y, cols, cv, data_label, max_depth=3, test_size=0.3):
        self.max_depth = max_depth
        self.test_size = test_size
        self.x = x
        self.y = y
        self.cols = cols
        self.data_label = data_label
        self.cv = cv

    def run(self):
        # load the red wine data
        # source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
        # source abstract: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
        #df = pd.read_csv('./winequality-red.csv', sep=';')
        
        
        # separate the x and y data
        # y = quality, x = all other features in the file
        #x, y = df.iloc[:, :11].values, df.iloc[:,11].values
        
        
        # test size now configurable
        # split the data into training and test data using 70:30 split
        # assign 30% of values to test
        
        # setting random_state so the split is reproducible for subsequent modeling iterations
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.test_size, random_state=0)
        
        
        # NOTE: I'm not going to pipeline the tree, though I could. In the other algorithms which
        # require scaling, I'll use a pipeline. One benefit of the tree model is there is no need
        # to perform feature scaling.
        '''
        pipe_tree = Pipeline([('scl', StandardScaler()),
                              ('clf', DecisionTreeClassifier(criterion='entropy', max_depth=self.max_depth, random_state=0))])
        '''
        
        
        # k-fold cross validation    
        # resample the test data without replacement. This means that each data point is part of a
        # test a training set only once. (paraphrased from Raschka p.176)
        # In Stratified KFold, the features are evenly disributed such that each test and training
        # set is an accurate representation of the whole training set.
        kfold = StratifiedKFold(y=y_train, n_folds=self.cv, random_state=0)        
        
        
        # do the cross validation
        scores = []
        for k, (train, test) in enumerate(kfold):
            # run the learning algorithm
            # Supported criteria are gini for the Gini impurity and entropy for the information gain.
            tree = DecisionTreeClassifier(criterion='entropy', max_depth=self.max_depth, random_state=0)
            #tree = DecisionTreeRegressor(criterion='mse', max_depth=self.max_depth, random_state=0)
            
            tree.fit(x_train[train], y_train[train])
            score = tree.score(x_train[test], y_train[test])
            scores.append(score)
            print('Fold:', k+1, ', Class dist.:', np.bincount(y_train[test]), 'Acc:', score)        
            

        # run the model on the whole training set
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
        #tree = DecisionTreeRegressor(criterion='mse', max_depth=self.max_depth, random_state=0)
        tree.fit(x_train, y_train)        
    
        
        # export a graphical representation of the tree
        dot_data = io.StringIO()
        export_graphviz(tree,
                        out_file=dot_data,
                        feature_names=self.cols)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        fn = self.data_label + '_graph.pdf'
        graph.write_pdf(fn)    

        
        # (Raschka, p.52)
        y_pred = tree.predict(x_test)
        ms = (y_test != y_pred).sum()
        print('Misclassified samples:', ms)
        
        # sklearn has a bunch of performance metrics, for example
        acc = accuracy_score(y_test, y_pred)
        print('Accuracy is', acc)
        
        # plot the learning curves using mlextend
        #plot_learning_curves(x_train, y_train, x_test, y_test, tree)
        #plt.show()
        
        
        # plot the learning curves using sklearn and matplotlib
        plt.clf()
        train_sizes, train_scores, test_scores = learning_curve(estimator=tree,
                                                                 X=x_train,
                                                                 y=y_train,
                                                                 cv=self.cv,
                                                                 n_jobs=-1)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes, train_mean,
                  color='blue', marker='o',
                  markersize=5,
                  label='training accuracy')
        
        plt.fill_between(train_sizes,
                         train_mean + train_std,
                         train_mean - train_std,
                         alpha=0.15, color='blue')
        
        plt.plot(train_sizes, test_mean,
                          color='green', marker='s',
                          markersize=5, linestyle='--',
                          label='validation accuracy')        

        plt.fill_between(train_sizes,
                         test_mean + test_std,
                         test_mean - test_std,
                         alpha=0.15, color='green')
        
        plt.grid()
        plt.title("Learning Curve with Decision Tree")
        plt.xlabel('Number of training samples')
        plt.ylabel('Accurancy')
        plt.legend(loc='lower right')
        fn = self.data_label + '_tree_learncurve.png'
        plt.savefig(fn)
        
        
        

        # plot the validation curves
        plt.clf()
        param_range = range(1,50)
        
        train_scores, test_scores = validation_curve(estimator=tree,
                                                     X=x_train,
                                                     y=y_train, 
                                                     param_name='max_depth', 
                                                     param_range=param_range, 
                                                     cv=self.cv,
                                                     n_jobs=-1)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(param_range, train_mean,
                 color='blue', marker='o',
                 markersize=5,
                 label='training accuracy')
        
        plt.fill_between(param_range,
                         train_mean + train_std,
                         train_mean - train_std,
                         alpha=0.15, color='blue')


        plt.plot(param_range, test_mean,
                 color='green', marker='s',
                 markersize=5, linestyle='--',
                 label='validation accuracy')
        
        plt.fill_between(param_range,
                         test_mean + test_std,
                         test_mean - test_std,
                         alpha=0.15, color='green')
        
        
        plt.grid()
        #plt.xscale('log')
        plt.title("Validation Curve with Decision Tree")
        plt.xlabel('Max Depth')
        plt.ylabel('Accurancy')
        plt.legend(loc='lower right')
        fn = self.data_label + '_tree_validationcurve.png'
        plt.savefig(fn)
        
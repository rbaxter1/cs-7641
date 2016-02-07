import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, Imputer
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline

from mlxtend.evaluate import plot_decision_regions, plot_learning_curves

import io
import pydotplus

from ggplot import *

from tree import *

# references:
# Raschka, Sebatian "Python Machine Learning"
    
# this method is left here for reference. I used it for testing and documenting steps 
def main_old():
    
    # 
    use_normalized = False #not used
    use_standardized = True
    
    # load the red wine data
    # source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
    # source abstract: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    df = pd.read_csv('./winequality-red.csv', sep=';')
    #print(df)
    
    # split the data into training and test data using 60:40 split
    # y = quality, x = all features
    #x, y = df.iloc[:, :11].values, df.iloc[:,11].values
    x, y = df.iloc[:,[1,10]].values, df.iloc[:,11].values
    
    # assign 30% of values to test
    # setting random_state so the split is reproducible for subsequent iterations
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
    
    # with decision trees and randoms we don't need to worry about feature scaling
    # every other algorithm will require feature scaling. we can either normalize
    # or standardize
    
    # using normalization we get a bounded interval
    if use_normalized:
        mms = MinMaxScaler
        x_train_norm = mms.fit_transform(x_train)
        x_test_norm = mms.fit_transform(x_test)
    
    # using standardization we often "center a feature around 0 with
    # a standard deviation of 1 so the feature takes on the form of
    # a normal distribution... furthermore standardization maintains
    # usefule information about the outliers" (Raschka, p. 111)
    # for logistic and svm use standardization
    if use_standardized:
        stdsc = StandardScaler()
        x_train = stdsc.fit_transform(x_train)
        x_test = stdsc.fit_transform(x_test)
    
    
    
    # k-fold cross validation    
    # resample the test data without replacement. This means that each data
    # point is part of a test a training set only once. (paraphrased from Raschka p.176)
    # In Stratified KFold, the features are evenly disributed such that each test and 
    # training set is an accurate representation of the whole
    kfold = StratifiedKFold(y=y_train,
                            n_folds=5,
                            random_state=0)
    
    # method 1: In this case we are not scaling any data, so the job
    # is simplified. If we needed to standardize the data, we would 
    # need to do so explicitly in each iteration. But sklean has a
    # method to simplify this using pipelining which is shown in 
    # the second method below.
    if not use_standardized:
        scores1 = []
        for k, (train, test) in enumerate(kfold):
            # run the learning algorithm
            # Supported criteria are gini for the Gini impurity and entropy for the information gain.
            tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
            tree.fit(x_train[train], y_train[train])
            score = tree.score(x_train[test], y_train[test])
            scores1.append(score)
            print('Fold:', k+1, ', Class dist.:', np.bincount(y_train[test]), 'Acc:', score)
        
    
    # method 2: use pipelining in the case where need to scale (or do dimensionality reduction)
    if use_standardized:
        pipe_tree = Pipeline([('scl', StandardScaler()),
                              ('clf', DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0))])
        
        # test it: this should match the non-pipelined call
        pipe_tree.fit(x_train, y_train)
        print('Test accuracy:', pipe_tree.score(x_test, y_test))
        
        # use with cross validation
        # one useful performance feature of using pipelines and cross validation is the option
        # to distribute the work across multiple CPUs using the n_jobs argument. -1 would use
        # all CPUs.
        scores2 = cross_val_score(estimator=pipe_tree,
                                  X=x_train,
                                  y=y_train,
                                  cv=5,
                                  n_jobs=-1)
        
        print('CV accuracy scores:', scores2)
        print('CV accuracy:', np.mean(scores2), '+/-', np.std(scores2))

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    tree.fit(x_train, y_train)        

    # (Raschka, p.52)
    y_pred = tree.predict(x_test)
    ms = (y_test != y_pred).sum()
    print('Misclassified samples:', ms)
    
    # sklearn has a bunch of performance metrics, for example
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy is', acc)
    
    # plot the learning curves for the pipeline and non-pipelined models
    plot_learning_curves(x_train, y_train, x_test, y_test, tree)
    plt.show()
    
    plot_learning_curves(x_train, y_train, x_test, y_test, pipe_tree)
    plt.show()

    
    # visualize the decision boundaries by looking at the rectangles which 
    # divide up the space.
    # TODO Would like to get this working
#    x_combined = np.vstack((x_train, x_test))
#    y_combined = np.hstack((y_train, y_test))
#    plot_decision_regions(X=x_combined, y=y_combined, clf=tree) #test_idx=range(105,150)
    #plot_decision_regions(x_train, y_train, clf=tree)
    #plt.xlabel('xxx')
    #plt.ylabel('quality')
#    plt.show()
    
    
    dot_data = io.StringIO()
    export_graphviz(tree,
                    out_file=dot_data,
                    feature_names=df.iloc[:, :11].columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree1.pdf")    

def main():
    
    
    #
    # DECISION TREE
    #
    
    #
    #
    # RED WINE
    #
    #    

    #
    # load the red wine data
    # source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
    # source abstract: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    df = pd.read_csv('./winequality-red.csv', sep=';')

    # separate the x and y data
    # y = quality, x = all other features in the file
    # update: using just volatile acid, ph and alcohol
    x, y = df.iloc[:,[1,8,10]].values, df.iloc[:,11].values    
    cols = df.iloc[:,[1,8,10]].columns

    myTree = rb_tree(x, y, cols, 'redwine')
    myTree.run()
    
    
    
    #
    #
    # WHITE WINE
    #
    #
    #df = pd.read_csv('./winequality-white.csv', sep=';')

    # separate the x and y data
    # y = quality, x = all other features in the file
    #x, y = df.iloc[:, :11].values, df.iloc[:,11].values    
    #cols = df.iloc[:, :11].columns

    #myTree = rb_tree(x, y, cols, 'whitewine')
    #myTree.run()    
    
    #
    # TITANIC
    #
    #
    # source: https://www.kaggle.com/c/titanic/data
    df = pd.read_csv('./titanic_train.csv', sep=',')
    
    
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
    
    #x, y = df.iloc[:,[2,10]].values, df.iloc[:,1].values
    cols = df.iloc[:,[0,1,3,4]].columns

    myTree = rb_tree(x, y, cols, 'titanic')
    myTree.run()    
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
'''
p = ggplot(aes(x='quality'), data=df) + \
    geom_histogram(binwidth=1) + ggtitle('Red Wine Quality') + labs('Quality', 'Freq')


ggplot(df, aes(x='density')) + \ 
    geom_histogram(aes(y=..density..), \
                   binwidth=.5, \
                   colour="black", fill="white") + \
    geom_density(alpha=.2, fill="#FF6666")


ggplot(aes(x='quality', y='alcohol', colour='quality'), data=df) + \
    geom_point()


p = ggplot(aes(x='quality', y='fixed acidity'), data=df)
p + geom_point(shape=1)

+ geom_smooth(method=lm, se=FALSE)

#ggplot(df, aes(x='fixed acidity', y='citric acid', color='quality')) +
#geom_point(shape=1) +
#geom_smooth()
    
    
ggplot(df, aes(x=xvar, y=yvar, color=cond)) +
    geom_point(shape=1) +
    scale_colour_hue(l=50) + # Use a slightly darker palette than normal
    geom_smooth(method=lm,   # Add linear regression lines
                se=FALSE)    # Don't add shaded confidence region
'''
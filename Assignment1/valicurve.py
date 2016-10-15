"""
==========================
Plotting Validation Curves
==========================

In this plot you can see the training scores and validation scores of an SVM
for different values of the kernel parameter gamma. For very low values of
gamma, you can see that both the training score and the validation score are
low. This is called underfitting. Medium values of gamma will result in high
values for both scores, i.e. the classifier is performing fairly well. If gamma
is too high, the classifier will overfit, which means that the training score
is good but the validation score is poor.
"""
#print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from data_helper import *



clf = pipeline = Pipeline([('scl', StandardScaler()),
                        ('clf', MLPClassifier(random_state=0,
                                              activation='relu',
                                              shuffle=True,
                                              solver='adam',
                                              learning_rate_init=0.001,
                                              learning_rate='constant',
                                              hidden_layer_sizes=(100,)
                                              ))])

#X, y = digits.data, digits.target
dh = data_helper()    
X, X_test, y, y_test =  dh.load_titanic_data_full_set()
    
param_range = np.arange(1, 10001, 1)

train_scores, test_scores = validation_curve(
    clf, X, y, param_name="clf__max_iter", param_range=param_range,
    cv=4, scoring="accuracy", n_jobs=1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fn = './output/' + str(uuid.uuid4()) + '_test.txt'
p = pandas.DataFrame(test_scores_mean)
p.to_csv(fn, header =False)

fn = './output/' + str(uuid.uuid4()) + '_train.txt'
p = pandas.DataFrame(train_scores_mean)
p.to_csv(fn, header =False)

#traintest = np.hstack((train_scores_mean, test_scores_mean))

plt.title("Validation Curve with ANN")
plt.xlabel("Iterations")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
fn = './output/' + str(uuid.uuid4()) + '_valicurve.png'
plt.savefig(fn)

#plt.show()

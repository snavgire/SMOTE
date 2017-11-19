__author__ = 'Sagar Navgire'

from sklearn.metrics import roc_curve, auc

# Reference: https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import tree
from scipy import interp
import pylab as pl

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from pprint import pprint
from sklearn.model_selection import GridSearchCV


import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.cross_validation import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

def plotROC(majoritySamples, minoritySamples):
    actual = [1, 1, 1, 0, 0, 0]
    predictions = [1, 1, 1, 0, 0, 0]

    allSamples = majoritySamples + minoritySamples

    # for i in range(0, len(allSamples)):
    #     print (allSamples[i])

    X = np.array(allSamples)
    data = X[:,:8]
    labels = X[:,8]

    labels = [int(i) for i in labels]

    print (labels)

    Y = np.unique(labels)
    print(Y)

    # print (np.unique(labels))
    # print (data[0])

    classifier = tree.DecisionTreeClassifier(max_depth=3)
    classifier = classifier.fit(data,labels)

    result = classifier.predict([['1', '85', '66', '29', '0', '26.6', '0.351', '31']])
    print(result)

    result2 = classifier.predict([['6', '148', '72', '35', '0', '33.6', '0.627', '50']])
    print (result2)

    # depth = []
    # for i in range(3, 20):
    #     clf = tree.DecisionTreeClassifier(max_depth=i)
    #     Perform 7-fold cross validation
        # scores = cross_val_score(estimator=clf, X=data, y=labels, cv=7, n_jobs=4)
        # depth.append((i, scores.mean()))
    # print(depth)

    # clf2 = tree.DecisionTreeClassifier
    # scores = cross_val_predict(classifier,data,labels, cv=10)
    # print (scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    param_grid = {'max_depth': np.arange(3, 10)}

    treeClassifier = GridSearchCV(DecisionTreeClassifier(), param_grid)

    cv = StratifiedKFold(labels, n_folds=6)
    for train, test in cv:
        treeClassifier.fit(data[train], labels[train])
        tree_preds = treeClassifier.predict_proba(data[test])[:, 1]
        tree_performance = roc_auc_score(labels[test], tree_preds)

        print 'DecisionTree: Area under the ROC curve = {}'.format(tree_performance)




    cv = StratifiedKFold(labels,n_folds=6)
    classifier2 = tree.DecisionTreeClassifier(max_depth=3)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv:
        probas_ = classifier2.fit(data[train], labels[train]).predict_proba(data[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(labels[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()



    # cv = StratifiedKFold(labels, n_folds=10)
    # classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
    #
    # mean_tpr = 0.0
    # mean_fpr = np.linspace(0, 1, 100)
    # all_tpr = []
    #
    # for i, (train, test) in enumerate(cv):
    #     probas_ = classifier.fit(X[train], labels[train]).predict_proba(X[test])
    #     # Compute ROC curve and area the curve
    #     fpr, tpr, thresholds = roc_curve(labels[test], probas_[:, 1])
    #     mean_tpr += interp(mean_fpr, fpr, tpr)
    #     mean_tpr[0] = 0.0
    #     roc_auc = auc(fpr, tpr)
    #     pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    #
    # pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    #
    # mean_tpr /= len(cv)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # pl.plot(mean_fpr, mean_tpr, 'k--',
    #         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    #
    # # pl.xlim([-0.05, 1.05])
    # # pl.ylim([-0.05, 1.05])
    # pl.xlabel('False Positive Rate')
    # pl.ylabel('True Positive Rate')
    # pl.title('Receiver operating characteristic example')
    # pl.legend(loc="lower right")
    # pl.show()

    # y_test = np.array(allSamples)[:, 8]
    # probas = np.array(allSamples)[:, 0]
    #
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probas)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    #
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(false_positive_rate, true_positive_rate, 'b',
    #          label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # # plt.xlim([-0.1, 1.2])
    # # plt.ylim([-0.1, 1.2])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig('Output/myfilename.png')
    # plt.show()
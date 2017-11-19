__author__ = 'Sagar Navgire'

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import tree

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from random import shuffle

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def treeClassifierLogisticRegression(majoritySamples, minoritySamples):
    allSamples = minoritySamples + majoritySamples

    nPArray = np.array(allSamples, dtype=float)
    data = nPArray[:, :8]
    labels = nPArray[:, 8]

    X = data
    y = [int(i) for i in labels]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)

    # instantiate model
    logreg = LogisticRegression()

    # fit model
    logreg.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = logreg.predict(X_test)

    print (type(y_test))
    print (type(y_pred_class))

    # print('True:', np.array(y_test).values[0:25])
    # print('False:', y_pred_class[0:25])

    print(metrics.accuracy_score(y_test, y_pred_class))
    # print (y_test.mean())
    print(metrics.confusion_matrix(y_test, y_pred_class))

    # save confusion matrix and slice into four pieces
    confusion = metrics.confusion_matrix(y_test, y_pred_class)

    # [row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    # use float to perform true division, not integer division
    print((TP + TN) / float(TP + TN + FP + FN))
    print(metrics.accuracy_score(y_test, y_pred_class))

    classification_error = (FP + FN) / float(TP + TN + FP + FN)

    print(classification_error)
    print(1 - metrics.accuracy_score(y_test, y_pred_class))

    sensitivity = TP / float(FN + TP)

    print(sensitivity)
    print(metrics.recall_score(y_test, y_pred_class))

    specificity = TN / (TN + FP)
    print(specificity)

    false_positive_rate = FP / float(TN + FP)

    print(false_positive_rate)
    print(1 - specificity)

    precision = TP / float(TP + FP)

    print(precision)
    print(metrics.precision_score(y_test, y_pred_class))

    print (logreg.predict(X_test)[0:10])
    print (logreg.predict_proba(X_test)[0:10])

    y_pred_prob = logreg.predict_proba(X_test)[:, 1]

    print("AUC: " + str(metrics.roc_auc_score(y_test, y_pred_prob)))

    print (cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean())

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for diabetes using C4.5 classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()

def treeClassifier(majoritySamples, minoritySamples):
    allSamples = minoritySamples + majoritySamples

    nPArray = np.array(allSamples, dtype=float)
    data = nPArray[:, :8]
    labels = nPArray[:, 8]

    X = data
    y = [int(i) for i in labels]

    clf = tree.DecisionTreeClassifier()
    # scores = cross_val_score(clf, X, y, cv=10)
    predicted = cross_val_predict(clf, X, y, cv=10)
    # print (scores.mean())
    # print (predicted)
    print (metrics.accuracy_score(y, predicted) )

    print("AUC: " + str(metrics.roc_auc_score(y, predicted)))

    print (cross_val_score(clf, X, y, cv=10, scoring='roc_auc').mean())

    fpr, tpr, thresholds = metrics.roc_curve(y, predicted)

    plt.plot(fpr * 100, tpr * 100)
    plt.xlim([0.0, 100.0])
    plt.ylim([0.0, 100.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for PIMA using C4.5 classifier')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.grid(True)
    plt.show()

def treeClassifier2(majoritySamples, minoritySamples):
    allSamples = minoritySamples + majoritySamples

    shuffle(allSamples)

    nPArray = np.array(allSamples)
    data = nPArray[:, :8]
    labels = nPArray[:, 8]

    X = data
    y = np.array(labels, dtype=int)

    kf = KFold(n_splits=10)
    clf = tree.DecisionTreeClassifier()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in kf.split(X, y):
        # X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        # clf = clf.fit(X_train, y_train)
        # print(clf.predict(X_test))
        # print (y_test)

        predicted = clf.fit(X[train], y[train]).predict(X[test])



        # Compute ROC curve and area the curve
        print("AUC[" + str(i) + "]: " + str(metrics.roc_auc_score(y[test], predicted)))
        print ("Accuracy: " + str(metrics.accuracy_score(y[test], predicted)))

        fpr, tpr, thresholds = roc_curve(y[test], predicted)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        # plt.plot(fpr * 100, tpr * 100)

        i += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # plt.plot(fpr, tpr, lw=1, alpha=0.3,
    #          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    # plt.plot(fpr * 100, tpr * 100)
    # plt.xlim([0.0, 100.0])
    # plt.ylim([0.0, 100.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for PIMA using C4.5 classifier')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.grid(True)
    plt.show()

def plotConvexHull():
    points = np.random.rand(4, 2)  # 30 random points in 2-D
    hull = ConvexHull(points)

    plt.plot(points[:, 0], points[:, 1], 'o')

    print (points[:, 0])
    print (points[:, 1])

    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    plt.show()

__author__ = 'Sagar Navgire'

import numpy as np
from sklearn.naive_bayes import GaussianNB
from random import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn import metrics
import matplotlib.pyplot as plt

def naiveBayes(majoritySamples, minoritySamples):
    allSamples = minoritySamples + majoritySamples

    shuffle(allSamples)

    nPArray = np.array(allSamples)
    data = nPArray[:, :8]
    labels = nPArray[:, 8]

    X = np.array(data, dtype=float)
    y = np.array(labels, dtype=int)

    kf = KFold(n_splits=10)
    clf = GaussianNB()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in kf.split(X, y):
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
    plt.title('ROC curve for PIMA using Naive Bayes')
    plt.xlabel('% False Positive')
    plt.ylabel('% True Positive')
    plt.grid(True)
    plt.show()


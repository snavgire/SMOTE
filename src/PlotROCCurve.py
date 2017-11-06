from sklearn.metrics import roc_curve, auc

# Reference: https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import random

def plotROC():
    actual = [1, 1, 1, 0, 0, 0]
    predictions = [1, 1, 1, 0, 0, 0]

    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('myfilename.png')
    plt.show()
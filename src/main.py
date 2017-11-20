__author__ = 'Sagar Navgire'

import csv
from smote import smote
from UnderSample import underSample
from PlotROCCurve import plotROC
from C4_5Tree import treeClassifier
from C4_5Tree import treeClassifier2
from C4_5Tree import plotConvexHull
from Helper import smoteHelper
from Helper import underSampleOnlyHelper
from NaiveBayes import naiveBayes
from Helper import getSeparatedSamples
from Helper import getSamples

if __name__ == '__main__':
    try:
        minorityCounter = 0
        majorityCounter = 0
        minoritySamples = []
        majoritySamples = []

        # smote.smote()


        # with open('Input/diabetes.csv', 'rb') as csvfile:
        #     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #     for row in spamreader:
        #         Sample = row[0].split(',')
        #         if (Sample[8] == '1'):
        #             minorityCounter += 1
        #             minoritySamples.append(Sample[0:9])
        #             # print (Sample[0:8])
        #
        #             # print(row[0][2])
        #             # print (', '.join(row))
        #         elif (Sample[8] == '0'):
        #             majorityCounter += 1
        #             majoritySamples.append(Sample[0:9])
        #

        minoritySamples, majoritySamples = getSeparatedSamples('Input/diabetes.csv')

        print ("Number of Miniority Samples:" + str(len(minoritySamples)))
        print ("Number of Majority Samples:" + str(len(majoritySamples)))
        # print minoritySamples[0]

        minorityCounter = len(minoritySamples)
        majorityCounter = len(majoritySamples)

        underSampledMajoritySamples = underSample(minorityCounter, 100, majoritySamples, majorityCounter)
        underSampleOnlyHelper(minoritySamples)

        smoteHelper(underSampledMajoritySamples)
        smote(minorityCounter, 100, 5, minoritySamples)

        # plotROC(majoritySamples, minoritySamples)
        # treeClassifierLogisticRegression(majoritySamples, minoritySamples)
        # treeClassifier(majoritySamples, minoritySamples)


        treeClassifier2('Output/diabetes_Smote.csv')
        treeClassifier2('Output/diabetes_Under.csv')
        # plotConvexHull()

        naiveBayes(majoritySamples, minoritySamples)

        # csvfile.close

    except Exception as error:
        print (error)

__author__ = 'Sagar Navgire'

import csv
from smote import smote
from UnderSample import underSample
from sklearn.neighbors import NearestNeighbors
import numpy as np

if __name__ == '__main__':
    try:
        minorityCounter = 0
        majorityCounter = 0
        minoritySamples = []
        majoritySamples = []

        # smote.smote()
        with open('Input/diabetes.csv', 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                Sample = row[0].split(',')
                if (Sample[8] == '1'):
                    minorityCounter += 1
                    minoritySamples.append(Sample[0:8])
                    # print (Sample[0:8])

                    # print(row[0][2])
                    # print (', '.join(row))
                elif (Sample[8] == '0'):
                    majorityCounter += 1
                    majoritySamples.append(Sample[0:9])
        print ("Number of Miniority Samples:" + str(minorityCounter))
        print ("Number of Majority Samples:" + str(majorityCounter))
        # print minoritySamples[0]

        # smote(minorityCounter, 300, 5, minoritySamples)
        underSample(minorityCounter, 100, majoritySamples, majorityCounter)

        csvfile.close

    except Exception as error:
        print (error)

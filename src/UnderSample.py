__author__ = 'Sagar Navgire'

import random
import csv


# T - Number of Minority Class Samples
# N - RAte of Undersampling

def underSample(T, N, majoritySamples, numberMajoritySamples):
    print ("In Under Sample")

    # if (N < 100):
    #     T = (N / 100) * T
    #     N = 100

    targetNumberMajoritySamples = int((T * 100) / N)
    print ("Number of target Majority Samples: " + str(targetNumberMajoritySamples))

    while (numberMajoritySamples > targetNumberMajoritySamples):
        indexToDelete = random.randint(0, numberMajoritySamples - 1)
        majoritySamples.remove(majoritySamples[indexToDelete])
        numberMajoritySamples -= 1

    print ("Number of Majority class: " + str(len(majoritySamples)))

    # fo2 = open("Output/diabetes_Smote.csv", "a")
    fo = open("Output/diabetes_Under.csv", "w+")

    for i in range(0, len(majoritySamples)):
        outputString = (
        str(majoritySamples[i][0]) + "," + str(majoritySamples[i][1]) + "," + str(majoritySamples[i][2]) + "," + str(
            majoritySamples[i][3]) + "," + str(majoritySamples[i][4]) + "," + str(majoritySamples[i][5]) + "," + str(
            majoritySamples[i][6]) + "," + str(majoritySamples[i][7]) + "," + str(majoritySamples[i][8]))

        # fo2.write(outputString + "\n")
        fo.write(outputString + "\n")
        print (outputString)

    # fo2.close()
    fo.close()
    return majoritySamples

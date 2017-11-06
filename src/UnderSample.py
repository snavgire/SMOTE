__author__ = 'Sagar Navgire'

import random


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

    print ("Number of Majjority class: " + str(len(majoritySamples)))

    for i in range(0, len(majoritySamples) - 1):
        print (majoritySamples[i])

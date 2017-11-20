__author__ = 'Sagar Navgire'

import csv

def getSamples(filename):
    allSamples = []

    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            Sample = row[0].split(',')
            allSamples.append(Sample[0:9])

    csvfile.close
    return allSamples

def getSeparatedSamples(filename):
    minoritySamples = []
    majoritySamples = []

    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            Sample = row[0].split(',')
            if (Sample[8] == '1'):
                # minorityCounter += 1
                minoritySamples.append(Sample[0:9])
                # print (Sample[0:8])

                # print(row[0][2])
                # print (', '.join(row))
            elif (Sample[8] == '0'):
                # majorityCounter += 1
                majoritySamples.append(Sample[0:9])

    csvfile.close
    return minoritySamples, majoritySamples


#Function to add majority samples to UnderSample Only file
def underSampleOnlyHelper(minoritySamples):
    fo = open("Output/diabetes_Under.csv", "a")

    for i in range(0, len(minoritySamples)):
        outputString = (
            str(minoritySamples[i][0]) + "," + str(minoritySamples[i][1]) + "," + str(
                minoritySamples[i][2]) + "," + str(
                minoritySamples[i][3]) + "," + str(minoritySamples[i][4]) + "," + str(
                minoritySamples[i][5]) + "," + str(
                minoritySamples[i][6]) + "," + str(minoritySamples[i][7]) + "," + str(minoritySamples[i][8]))

        fo.write(outputString + "\n")
        print (outputString)

    fo.close()

# Function to add under samples majority samples to smote file
def smoteHelper(underSampledMajoritySamples):
    fo2 = open("Output/diabetes_Smote.csv", "w+")

    for i in range(0, len(underSampledMajoritySamples)):
        outputString = (
            str(underSampledMajoritySamples[i][0]) + "," + str(underSampledMajoritySamples[i][1]) + "," + str(
                underSampledMajoritySamples[i][2]) + "," + str(
                underSampledMajoritySamples[i][3]) + "," + str(underSampledMajoritySamples[i][4]) + "," + str(
                underSampledMajoritySamples[i][5]) + "," + str(
                underSampledMajoritySamples[i][6]) + "," + str(underSampledMajoritySamples[i][7]) + "," + str(underSampledMajoritySamples[i][8]))

        fo2.write(outputString + "\n")

    fo2.close()

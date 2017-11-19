__author__ = 'Sagar Navgire'

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

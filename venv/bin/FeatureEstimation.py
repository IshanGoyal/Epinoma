'''
Filename: FeatureEstimation.py
Objective: Using the random forest function, feature estimator comes up with a weighted list of features from a given number of runs
Input: Number of times to run the random forest algorithm
Output: Dictionary of features sorted by their weighted feature importance
'''


from RandomForestFeatureList import featurelist
import operator

def estimateFeatures(numberofRandomForests):
    allFeatures = {}
    numCycles = numberofRandomForests
    #Run random forest for numCycles
    for x in range(0,numCycles):
        print("INDEX: " + str(x))
        #Run the program with 10 decision trees in each random forest
        result = list(featurelist(10,30))
        #Iterate through all the top 30 features from the given random forest
        for item in result:
            #If the feature is already in the dictionary, add to its weight
            if item[0] in allFeatures:
                allFeatures[str(item[0])] += float(item[1])
            #If the feature hasn't been seen before, add it to the dictionary
            else:
                allFeatures[str(item[0])] = float(item[1])
        print(allFeatures)

    for key in allFeatures.keys(): #normalize the dictionary by the number of cycles
        allFeatures[key] = float(allFeatures[key]/numCycles)
    #sort the dictionary by the second value in the tuple in reverse order
    sorted_dict = sorted(allFeatures.items(), key=operator.itemgetter(1), reverse=True)[:30]
    return sorted_dict

output = estimateFeatures(50)
print(output)

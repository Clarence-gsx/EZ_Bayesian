import numpy as np
import pandas as pd
from functools import reduce



## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed 
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs
    return factorTable


## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]
    # 得到所有列的取值

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)

    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities 
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i, 'probs'] = sum(a == (i + 1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j, 'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


## Join of two factors
## factor1, factor2: two factor tables
##
## Should return a factor table that is the join of factor 1 and 2.
## You can assume that the join of two factors is a valid operation.
## Hint: You can look up pd.merge for mergin two factors
def joinFactors(factor1, factor2):
    key = [x for x in list(factor1) if x in list(factor2)]
    if 'probs' in key:
        key.remove('probs')
    if key:
        factor3 = pd.merge(factor1, factor2, how='outer', on=key, suffixes=['_l', '_r'])
        factor3['probs'] = factor3['probs_l'] * factor3['probs_r']
        factor3.drop(['probs_l', 'probs_r'], axis=1, inplace=True)
        return factor3
    else:
        orderkey1 = factor1.shape[0] * [1]
        orderkey2 = factor2.shape[0] * [1]
        factor1.insert(1, 'orderkey', orderkey1)
        factor2.insert(1, 'orderkey', orderkey2)
        factor3 = pd.merge(factor1, factor2, how='outer', on='orderkey', suffixes=['_l', '_r'])
        factor3['probs'] = factor3['probs_l'] * factor3['probs_r']
        factor3.drop(['probs_l', 'probs_r', 'orderkey'], axis=1, inplace=True)
        factor1.drop('orderkey', axis=1, inplace=True)
        factor2.drop('orderkey', axis=1, inplace=True)
        return factor3


## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the left side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    hiddenVar = [x for x in list(factorTable) if x in hiddenVar]
    if not hiddenVar:
        print('invalid hiddenVar!')
        return factorTable
    key = [x for x in list(factorTable) if x not in hiddenVar]
    if 'probs' in key:
        key.remove('probs')
    if key:
        result = factorTable.groupby(key).sum()
        result.drop(hiddenVar, axis=1, inplace=True)
        result = result.reset_index()
        return result
    else:
        print('Nothing left for summing the hiddenVar!')
        return factorTable


## Marginalize a list of variables
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
##
## Should return a Bayesian network containing a list of factor tables that results
## when the list of variables in hiddenVar is marginalized out of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
    leftFactor = bayesNet
    for var in hiddenVar:
        joinFactor=[x for x in leftFactor if var in list(x)]
        leftFactor=[x for x in leftFactor if var not in list(x)]
        if joinFactor:
            temp_factor = joinFactor[0]
            for net in joinFactor[1:]:
                temp_factor=joinFactors(net,temp_factor)
            temp_factor=marginalizeFactor(temp_factor,var)
            leftFactor.append(temp_factor)
    return leftFactor


## Update BayesNet for a set of evidence variables
## bayesNet: a list of factor and factor tables in dataframe format
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesNet, evidenceVars, evidenceVals):
    temp=[x.copy(deep=True) for x in bayesNet]
    for i in range(len(evidenceVars)):
        for j in range(len(bayesNet)):
            if evidenceVars[i] in list(temp[j]):
                temp[j] = temp[j][temp[j][evidenceVars[i]].isin([evidenceVals[i]])]
                temp[j] = temp[j].reset_index(drop=True)
    return temp


## Run inference on a Bayesian network
## bayesNet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using 
## join and marginalization of the sets of variables. 
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesNet, hiddenVar, evidenceVars, evidenceVals):
    Net=evidenceUpdateNet(bayesNet,evidenceVars,evidenceVals)
    factorlist = marginalizeNetworkVariables(Net,hiddenVar)
    result = factorlist[0]
    for factor in factorlist[1:]:
        result=joinFactors(result,factor)
    result['probs']=result['probs']/result['probs'].sum()
    return result


#!/usr/bin/env python3

from BayesianNetworks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#############################
## Example Tests from Bishop Pattern recognition textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF] # carNet is a list of factors
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)

marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, ['fuel'], [1])
evidenceUpdateNet(carNet, ['fuel', 'battery'], [1, 0])

## Marginalize must first combine all factors involving the variable to
## marginalize. Again, this operation may lead to factors that aren't
## probabilities.
marginalizeNetworkVariables(carNet, ['battery']) ## this returns back a list
marginalizeNetworkVariables(carNet, ['fuel']) ## this returns back a list
marginalizeNetworkVariables(carNet, ['battery', 'fuel'])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []) )        ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))           ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))          ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))    ## chapter 8 equation (8.33)
print("inference ends")
###########################################################################
#RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv('RiskFactorsData.csv')

# Create factors

income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])
## you need to create more factor tables

risk_net = [income, smoke, exercise, bmi, diabetes]
print("income dataframe is ")
print(income)
factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2)

margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise'})
obsVars  = ['smoke', 'exercise']
obsVals  = [1, 2]

p = inference(risk_net, margVars, obsVars, obsVals)
print(p)


### Please write your own test scrip similar to  the previous example
###########################################################################
#HW4 test scrripts start from here
###########################################################################
# Create factors
income1 = readFactorTablefromData(riskFactorNet, ['income'])
smoke1 = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise1 = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
bmi1 = readFactorTablefromData(riskFactorNet, ['bmi', 'income', 'exercise'])
diabetes1 = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])
bp1 = readFactorTablefromData(riskFactorNet, ['bp', 'exercise', 'income', 'smoke'])
cholesterol1 = readFactorTablefromData(riskFactorNet, ['cholesterol', 'smoke', 'income', 'exercise'])
stroke1 = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi','bp', 'cholesterol'])
attack1 = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol'])
angina1 = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol'])
# you need to create more factor tables


Net1 = [income1, smoke1, exercise1, bmi1, diabetes1, bp1, cholesterol1, stroke1, attack1, angina1]
# Q1
print("Question 1:")
q1_1=0
for i in Net1:
    q1_1+=len(i)
print("size of this network is:",q1_1)
q1_2=len(inference(Net1,[],[],[]))
print("number of probabilities needed to store the full joint distribution is:",q1_2)

# Q2_a
print("Question 2:")
obsVars_q2_1 = ['smoke', 'exercise']
obsVals_q2_1 = [1, 2]
obsVals_q2_2 = [2, 1]
margVars_q2a_1 = list(set(factors) - {'diabetes', 'smoke', 'exercise'})
margVars_q2a_2= list(set(factors) - {'angina', 'smoke', 'exercise'})
margVars_q2a_3 = list(set(factors) - {'stroke', 'smoke', 'exercise'})
margVars_q2a_4= list(set(factors) - {'attack', 'smoke', 'exercise'})
print("bad habits:")
print(inference(Net1,margVars_q2a_1,obsVars_q2_1,obsVals_q2_1))
print(inference(Net1,margVars_q2a_2,obsVars_q2_1,obsVals_q2_1))
print(inference(Net1,margVars_q2a_3,obsVars_q2_1,obsVals_q2_1))
print(inference(Net1,margVars_q2a_4,obsVars_q2_1,obsVals_q2_1))
print("good habits:")
print(inference(Net1,margVars_q2a_1,obsVars_q2_1,obsVals_q2_2))
print(inference(Net1,margVars_q2a_2,obsVars_q2_1,obsVals_q2_2))
print(inference(Net1,margVars_q2a_3,obsVars_q2_1,obsVals_q2_2))
print(inference(Net1,margVars_q2a_4,obsVars_q2_1,obsVals_q2_2))

# Q2_b
obsVars_q2_b = ['bp', 'cholesterol','bmi']
obsVals_q2_b1= [1, 1,3]
obsVals_q2_b2=[3,2,2]
margVars_q2b_1 = list(set(factors) - {'diabetes', 'bp', 'cholesterol','bmi'})
margVars_q2b_2 = list(set(factors) - {'angina', 'bp', 'cholesterol','bmi'})
margVars_q2b_3 = list(set(factors) - {'stroke', 'bp', 'cholesterol','bmi'})
margVars_q2b_4 = list(set(factors) - {'attack', 'bp', 'cholesterol','bmi'})
print("bad health")
print(inference(Net1,margVars_q2b_1,obsVars_q2_b,obsVals_q2_b1))
print(inference(Net1,margVars_q2b_2,obsVars_q2_b,obsVals_q2_b1))
print(inference(Net1,margVars_q2b_3,obsVars_q2_b,obsVals_q2_b1))
print(inference(Net1,margVars_q2b_4,obsVars_q2_b,obsVals_q2_b1))
print("good health")
print(inference(Net1,margVars_q2b_1,obsVars_q2_b,obsVals_q2_b2))
print(inference(Net1,margVars_q2b_2,obsVars_q2_b,obsVals_q2_b2))
print(inference(Net1,margVars_q2b_3,obsVars_q2_b,obsVals_q2_b2))
print(inference(Net1,margVars_q2b_4,obsVars_q2_b,obsVals_q2_b2))

# Q3
print("Question 3:")
margVars_q3_1 = list(set(factors) - {'angina','income'})
margVars_q3_2 = list(set(factors) - {'stroke','income'})
margVars_q3_3 = list(set(factors) - {'attack','income'})
margVars_q3_4 = list(set(factors) - {'diabetes','income'})
q3_1=[]
q3_2=[]
q3_3=[]
q3_4=[]
x=[i for i in range(1,9)]
for i in range(1,9):
    p = evidenceUpdateNet([inference(Net1, margVars_q3_1, ['income'], [i])],["angina"],[1])[0]
    q3_1.append(p['probs'][0])
for i in range(1,9):
    p = evidenceUpdateNet([inference(Net1, margVars_q3_2, ['income'], [i])],["stroke"],[1])[0]
    q3_2.append(p['probs'][0])
for i in range(1,9):
    p = evidenceUpdateNet([inference(Net1, margVars_q3_3, ['income'], [i])],["attack"],[1])[0]
    q3_3.append(p['probs'][0])
for i in range(1,9):
    p = evidenceUpdateNet([inference(Net1, margVars_q3_4, ['income'], [i])],["diabetes"],[1])[0]
    q3_4.append(p['probs'][0])
plt.subplot(221)
plt.bar(x,q3_1)
plt.xlabel("income")
plt.ylabel("probability")
plt.title("P(angina=1|income=i)")
plt.subplot(222)
plt.bar(x,q3_2)
plt.xlabel("income")
plt.ylabel("probability")
plt.title("P(stroke=1|income=i)")
plt.subplot(223)
plt.bar(x,q3_3)
plt.xlabel("income")
plt.ylabel("probability")
plt.title("P(attack=1|income=i)")
plt.subplot(224)
plt.bar(x,q3_4)
plt.xlabel("income")
plt.ylabel("probability")
plt.title("P(diabetes=1|income=i)")
plt.tight_layout()
plt.show()


#Q4
income2 = readFactorTablefromData(riskFactorNet, ['income'])
smoke2 = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise2 = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
bmi2 = readFactorTablefromData(riskFactorNet, ['bmi', 'income', 'exercise'])
diabetes2 = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi','smoke','exercise'])
bp2 = readFactorTablefromData(riskFactorNet, ['bp', 'exercise', 'income', 'smoke'])
cholesterol2 = readFactorTablefromData(riskFactorNet, ['cholesterol', 'smoke', 'income', 'exercise'])
stroke2 = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi','bp', 'cholesterol','smoke','exercise'])
attack2 = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol','smoke','exercise'])
angina2 = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol','smoke','exercise'])

Net2=[income2,smoke2,exercise2,bmi2,diabetes2,bp2,cholesterol2,stroke2,attack2,angina2]
#a
print("Question 4:")
obsVars_q4_1 = ['smoke', 'exercise']
obsVals_q4_1 = [1, 2]
obsVals_q4_2 = [2, 1]
margVars_q4a_1 = list(set(factors) - {'diabetes', 'smoke', 'exercise'})
margVars_q4a_2= list(set(factors) - {'angina', 'smoke', 'exercise'})
margVars_q4a_3 = list(set(factors) - {'stroke', 'smoke', 'exercise'})
margVars_q4a_4= list(set(factors) - {'attack', 'smoke', 'exercise'})
print("bad habits:")
print(inference(Net2,margVars_q4a_1,obsVars_q4_1,obsVals_q4_1))
print(inference(Net2,margVars_q4a_2,obsVars_q4_1,obsVals_q4_1))
print(inference(Net2,margVars_q4a_3,obsVars_q4_1,obsVals_q4_1))
print(inference(Net2,margVars_q4a_4,obsVars_q4_1,obsVals_q4_1))
print("good habits:")
print(inference(Net2,margVars_q4a_1,obsVars_q4_1,obsVals_q4_2))
print(inference(Net2,margVars_q4a_2,obsVars_q4_1,obsVals_q4_2))
print(inference(Net2,margVars_q4a_3,obsVars_q4_1,obsVals_q4_2))
print(inference(Net2,margVars_q4a_4,obsVars_q4_1,obsVals_q4_2))

# b
obsVars_q4_b = ['bp', 'cholesterol','bmi']
obsVals_q4_b1= [1, 1,3]
obsVals_q4_b2=[3,2,2]
margVars_q4b_1 = list(set(factors) - {'diabetes', 'bp', 'cholesterol','bmi'})
margVars_q4b_2 = list(set(factors) - {'angina', 'bp', 'cholesterol','bmi'})
margVars_q4b_3 = list(set(factors) - {'stroke', 'bp', 'cholesterol','bmi'})
margVars_q4b_4 = list(set(factors) - {'attack', 'bp', 'cholesterol','bmi'})
print("bad health")
print(inference(Net2,margVars_q4b_1,obsVars_q4_b,obsVals_q4_b1))
print(inference(Net2,margVars_q4b_2,obsVars_q4_b,obsVals_q4_b1))
print(inference(Net2,margVars_q4b_3,obsVars_q4_b,obsVals_q4_b1))
print(inference(Net2,margVars_q4b_4,obsVars_q4_b,obsVals_q4_b1))
print("good health")
print(inference(Net2,margVars_q4b_1,obsVars_q4_b,obsVals_q4_b2))
print(inference(Net2,margVars_q4b_2,obsVars_q4_b,obsVals_q4_b2))
print(inference(Net2,margVars_q4b_3,obsVars_q4_b,obsVals_q4_b2))
print(inference(Net2,margVars_q4b_4,obsVars_q4_b,obsVals_q4_b2))


#Q5
# a
print("Question 5:")
print("Before adding the edge")
obsVars_q5 = ['diabetes']
obsVals_q5_1= [1]
obsVals_q5_2= [3]
margVars_q5 = list(set(factors) - {'diabetes', 'stroke'})
print(evidenceUpdateNet([inference(Net2,margVars_q5,obsVars_q5,obsVals_q5_1)],['stroke'],[1])[0])
print(evidenceUpdateNet([inference(Net2,margVars_q5,obsVars_q5,obsVals_q5_2)],['stroke'],[1])[0])
print("After adding the edge")
# b
stroke3 = readFactorTablefromData(riskFactorNet, ['stroke', 'diabetes','bmi','bp', 'cholesterol','smoke','exercise'])
Net3=[income2,smoke2,exercise2,bmi2,diabetes2,bp2,cholesterol2,stroke3,attack2,angina2]
print(evidenceUpdateNet([inference(Net3,margVars_q5,obsVars_q5,obsVals_q5_1)],['stroke'],[1])[0])
print(evidenceUpdateNet([inference(Net3,margVars_q5,obsVars_q5,obsVals_q5_2)],['stroke'],[1])[0])




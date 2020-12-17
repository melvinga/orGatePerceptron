###############################################################################################################################################################
# Filename    : orGatePerceptron.py
# Author      : Melvin George
# Created at  : 17 Dec 2020
# Last updated: 17 Dec 2020
# Description : Perceptron is a elementary neuron without hidden layers like neural networks.
#               It does not exist in the real world, but helps us to understand how machine learning happens.
#               This code tries to show how a perceptron learns to predict an OR gate.
#               It uses gradient descent algorithm to tune the weightages for inputs.
# Reference   : https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf
# Note        : This is an independent implementation of what is said in the reference content above.
#               Aim is to make the code readable and understandable by novice programmers. So, it is not optimized for speed in computation.
#               To see an excel sheet implementation of the same logic, open the orGatePerceptron.ods in LibreOffice Calc (application in Ubuntu/Linux).               
###############################################################################################################################################################

import numpy as np

# USER EDITABLE VALUES
lr = 0.5 # learning rate
         # larger values = faster learning, but very inaccurate
         # smaller value = slower learning, more accurate
MSE_EXPECTED = 1 / (1 * 1000 * 1000)
         # MSE = mean squared error
         # Kind of determines how much accuracy should the perceptron attain before it stops learning further
ITR_MULT_TO_PRINT_INFO = 1000
         # Every 1000th iteration result is printed. The other 999 not printed.
         # Printing an iteration resutlt makes the program slower. Every iteration result it skips printing makes it that much faster.

# DO NOT EDIT THE VALUES AND CODE BELOW. Just kidding! Edit and have fun :)
# OR gate - 2 inputs, 1 output
inputA = (0, 0, 1, 1)
inputB = (0, 1, 0, 1)
target = (0, 1, 1, 1)

# starting values for weights and bias
w1 = 0.15
w2 = 0.25
w3 = 0.35
bias = 1

# temporary storage for an iteration
mseTotal = 0.0

def sigmoid(x):
    val = 1 / (1 + np.exp(-x))
    return val

def inputAgg(inA, inB, bias, w1, w2, w3):
    val = (w1*inA) + (w2*inB) + (w3*bias)
    return val

def predicted(x):
    val = sigmoid(x)
    return val

def meanSqErrTerm(to, po):
    d = to - po
    val = 0.5 * d * d
    return val

def partialErrorOverOutput(predictedOutput, target):
    val = predictedOutput - target
    return val

def partialOutputOverInput(predictedOutput):
    val = predictedOutput * (1 - predictedOutput)
    return val

def partialInputOverWeight(input):
    val = input
    return val

def partialErrorOverWeight(fr1, fr2, fr3):
    val = fr1 * fr2 * fr3
    return val

def newValueOfWeight(w, lr, fr):
    val = w - (lr * fr)
    return val

def oneIteration(inputA, inputB, bias, w1, w2, w3, lr):
    w1new = 0.0
    w2new = 0.0
    w3new = 0.0
    mse = 0.0
    for i in range(len(target)):
        inA = inputA[i]
        inB = inputB[i]
        tOut = target[i]

        ia = inputAgg(inA, inB, bias, w1, w2, w3) # input aggregate
        po = predicted(ia) # predicted output
        mseTerm = meanSqErrTerm(tOut, po) # term to add to mse of whole iteration
        
        dErrorOverOutput = partialErrorOverOutput(po, tOut)
        dOutputOverInput = partialOutputOverInput(po)
        dInputOverWeight = 0
        if i==0:
            dInputOverWeight = partialInputOverWeight(bias)
        elif i==1:
            dInputOverWeight = partialInputOverWeight(inB)
        elif i==2:
            dInputOverWeight = partialInputOverWeight(inA)
        else:
            dInputOverWeight = partialInputOverWeight(inA) # or inB. Doesn't matter which one.

        dErrorOverWeight = partialErrorOverWeight(  dErrorOverOutput, \
                                                    dOutputOverInput, \
                                                    dInputOverWeight)
        
        newWeight = 0
        if i==0:
            newWeight = newValueOfWeight(w3, lr, dErrorOverWeight)
        elif i==1:
            newWeight = newValueOfWeight(w2, lr, dErrorOverWeight)
        elif i==2:
            newWeight = newValueOfWeight(w1, lr, dErrorOverWeight)
        else:
            newWeight = 0

        if i==0:
            w3new = newWeight
        elif i==1:
            w2new = newWeight
        elif i==2:
            w1new = newWeight
        else:
            newWeight = newWeight

        mse += mseTerm

    return w1new, w2new, w3new, mse

def displayInfo(itrCount, w1prev, w1, w2prev, w2, w3prev, w3, mseTotal):
    print("\n")
    print("iteration : ",itrCount)
    print("w1 prev   : ",w1prev)
    print("w1 new    : ",w1)
    print("w2 prev   : ",w2prev)
    print("w2 new    : ",w2)
    print("w3 prev   : ",w3prev)
    print("w3 new    : ",w3)
    print("mseTotal  : ",mseTotal)

doLoop = True
itrCount = 0
w1prev = 0.0
w2prev = 0.0
w3prev = 0.0
while doLoop:
    itrCount += 1
    w1prev = w1
    w2prev = w2
    w3prev = w3

    # Calculate new set of weights and mse
    w1, w2, w3, mseTotal = oneIteration(inputA, inputB, bias, w1, w2, w3, lr)
    
    # Choose whether progress should be updated to user in this iteration
    printInfo = False
    if ITR_MULT_TO_PRINT_INFO < 1000:
        ITR_MULT_TO_PRINT_INFO = 1000
    if itrCount > (ITR_MULT_TO_PRINT_INFO * 10):
        if (itrCount % (ITR_MULT_TO_PRINT_INFO * 10)) == 0:
            printInfo = True
        else:
            printInfo = False
    elif itrCount > ITR_MULT_TO_PRINT_INFO:
        if (itrCount % ITR_MULT_TO_PRINT_INFO) == 0:
            printInfo = True
        else:
            printInfo = False
    elif itrCount > (ITR_MULT_TO_PRINT_INFO / 10):
        if (itrCount % (ITR_MULT_TO_PRINT_INFO / 10)) == 0:
            printInfo = True
        else:
            printInfo = False
    else:
        printInfo = True

    # Update progress to user
    if printInfo:
        displayInfo(itrCount, w1prev, w1, w2prev, w2, w3prev, w3, mseTotal)

    # Check if one more iteration is needed
    if mseTotal < MSE_EXPECTED:
        doLoop = False

# Display final iteration count and weights obtained
displayInfo(itrCount, w1prev, w1, w2prev, w2, w3prev, w3, mseTotal)

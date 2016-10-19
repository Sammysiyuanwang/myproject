# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:37:35 2016

@author: wang
"""
#Logistic Regression 

import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(InX):
    return 1.0 / (1 + np.exp(-InX))
    
def trainLogisticRegression(train_x, train_y, alpha = 0.01, maxIter = 200, optimizeType = 'stocGradDescent'):
    #calculate training time
    startTime = time.time()
    
    numSamples, numFeatures = np.shape(train_x)   
    weights = np.ones((numFeatures, 1))
    
    #optimize the algorithm
    for k in range(maxIter):
        #gradient descent algorilthm
        if optimizeType == 'gradDescent':
            output = sigmoid(train_x * weights)
            error = train_y - output 
            weights = weights + alpha * train_x.transpose() * error
        #stochastic gradient descent   
        elif optimizeType == 'stocGradDescent':
            for i in range(numSamples):
                output = sigmoid(train_x[i,:] * weights)
                error = train_y[i,0] - output
                weights = weights + alpha * train_x[i,:].transpose() * error
                
        else:
            raise NameError('Not support optimize method type!')  
            
    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)  
    return weights
    
def predictLR(weights, input_x):
    outputs = sigmoid(input_x * weights)
#    output = []
#    for i in range(len(outputs)):
#        if outputs[i][0] >= 0.5:      
#            y = 1
#        else:
#            y = 0
#        output.append(y)
    output = np.round(outputs)
    return output
    
#load data from txt and convert to matrix
def loadData():  
    train_x = []  
    train_y = []  
    fileIn = open('testSet.txt')  
    for line in fileIn.readlines():  
        lineArr = line.strip().split()  
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])  
        train_y.append(float(lineArr[2]))  
    return np.mat(train_x), np.mat(train_y).transpose()  

#calculate the accuracy of the prediction
def Accuracy(predict_y, validate_y):
    if len(predict_y) == len(validate_y):
        matchCount = 0
        for i in range(len(predict_y)):
            if predict_y[i][0] == validate_y[i][0]:
                matchCount += 1
        accuracy = float(matchCount) / len(validate_y)
        return accuracy
    else:
        print 'The elements of the list don\'t match '

# show your trained logistic regression model only available with 2-D data  
def showLogRegres(weights, train_x, train_y):
    #train_x and train_y must be matrix datatype
    numSamples, numFeatures = np.shape(train_x)
    if numFeatures != 3:
        print 'Sorry!We can only plot 2D data!'
        return 1
    
    #draw all samples
    for k in xrange(numSamples):
        if int(train_y[k, 0]) == 0:
            plt.plot(train_x[k, 1], train_x[k, 2], 'ro')
        elif int(train_y[k, 0]) == 1:
            plt.plot(train_x[k, 1], train_x[k, 2], 'bo')
    
    #draw the classify line:
    min_x = min(train_x[:, 1])[0, 0]  
    max_x = max(train_x[:, 1])[0, 0]  
    weights = weights.getA()  # convert mat to array  
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]  
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]  
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  
    plt.xlabel('X1'); plt.ylabel('X2')  
    plt.show()  
        
    

if __name__ == '__main__':
    train_x, train_y = loadData()
    test_x = train_x; test_y = train_y 
    
    lr = trainLogisticRegression(train_x, train_y)
    y = predictLR(lr, test_x)
    a = Accuracy(y, test_y)
    y = y.astype(int)
    y.tolist()
    print y, '\n',a
    showLogRegres(lr, train_x, train_y)

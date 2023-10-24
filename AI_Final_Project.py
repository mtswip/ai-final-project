import math#used to get euler's number for activation functions
from sklearn import datasets#used to get dataset
import random#used to initialize weights

def sigG(val):#sigmoid function
    temp1 = (1 + (math.e ** (0 - val)))

    if (temp1 == 0):#this check prevents zero division errors by returning the value itself instead
        return val
    
    else:
        return (1 / temp1)

def linG(val):#this doesn't really need to be its own function, but is for the sake of consistency
    return val

def softMax(arr):#softmax activation function was found to reduce overfitting
    sum = 0.0

    for i in range(len(arr)):
        sum += (math.e ** arr[i])
    
    for i in range(3):
        arr[i] = ((math.e ** arr[i]) / sum)
    
    return arr

def neuron(xList, weights, bias, activation):#basic McCulloch-Pitts neuron
    sum = 0.0
    
    for i in range(len(xList)):
        sum += (xList[i] * weights[i])
    sum += bias

    match activation:        
        case "sigmoid":
            return sigG(sum)
        
        case "linear":
            return linG(sum)

def forwardPass(xList, weights, biases):#forward pass through the neural network
    h = [0.0, 0.0, 0.0, 0.0]
    y = [0.0, 0.0, 0.0]
    
    h[0] = neuron(xList, weights[0:4], biases[0], "sigmoid")#these first four neurons are the hidden layer
    h[1] = neuron(xList, weights[4:8], biases[1], "sigmoid")#I tested several activation functions for this layer, but sigmoid proved the best
    h[2] = neuron(xList, weights[8:12], biases[2], "sigmoid")
    h[3] = neuron(xList, weights[12:16], biases[3], "sigmoid")

    y[0] = neuron(h, weights[16:20], biases[4], "linear")#this is the output layer
    y[1] = neuron(h, weights[20:24], biases[5], "linear")#linear activation allows for the full output layer to be put through softmax function
    y[2] = neuron(h, weights[24:28], biases[6], "linear")

    y = softMax(y)#softmax seems to result in less overfitting

    if (max(y) == y[0]):#this chooses the result based on the neuron with the greatest output
        return [0, h, y]
    
    elif (max(y) == y[1]):
        return [1, h, y]
    
    elif (max(y) == y[2]):
        return [2, h, y]
    
    else:#this case should not occur, but it exists just in case
        return [-1, h, y]

def backpropagation(iris, weights, biases, learnRate):
    allX = iris.data[:, :4]#initializing variables
    trainingData = []
    validationData = []
    teacherNums = iris.target
    trainingTeacherNums = []
    validationTeacherNums = []
    error = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    deltaWeights = []
    deltaBiases = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    trainingAccuracy = 0.0
    validationAccuracy = 0.0

    for i in range(150):#setting training vs validation data
        if ((i % 5) < 3):
            trainingData.append(allX[i])
            trainingTeacherNums.append(teacherNums[i])

        elif ((i % 5) == 3):
            validationData.append(allX[i])
            validationTeacherNums.append(teacherNums[i])

    for i in range(28):#setting up deltaWeights because 28 is too many values to intialize in one line
        deltaWeights.append(0.0)

    for i in range(90):#training on each training value
        xList = trainingData[i]
        teacher = [0.0, 0.0, 0.0]

        match trainingTeacherNums[i]:#these values correspond to the output neurons, and are used in the output error calculation step
            case 0:
                teacher[0] = 1.0
            
            case 1:
                teacher[1] = 1.0
            
            case 2:
                teacher[2] = 1.0
        
        output = forwardPass(xList, weights, biases)#forward pass through the network with the current datapoint

        if (output[0] == trainingTeacherNums[i]):#increases accuracy stat and sets all errors and weight/bias changes to 0 if the model guesses correctly
            trainingAccuracy += 1.0
            for j in range(3):
                error[j + 4] = 0.0
                deltaBiases[j + 4] = 0.0
                for k in range(4):
                    deltaWeights[j + k + 16] = 0.0
        
        else:
            for j in range(3):#setting error and weight/bias changes for output units
                error[j + 4] = output[2][j] * (1 - output[2][j]) * (teacher[j] - output[2][j])
                deltaBiases[j + 4] = learnRate * error[j + 4]
                for k in range(4):
                    deltaWeights[j + k + 16] = learnRate * output[1][k] * error[j + 4]

        for j in range(4):#setting error and weight/bias changes for hidden units
            weightsXErrorSum = 0.0
            for k in range(3):
                weightsXErrorSum += (weights[j + k*4 + 16] * error[k + 4])
            
            error[j] = output[1][j] * (1 - output[1][j]) * weightsXErrorSum
            deltaBiases[j] = learnRate * error[j]
            for k in range(4):
                deltaWeights[j + k] = learnRate * xList[k] * error[j]
        
        for j in range(28):#updating weights
            oldWeight = weights[j]
            weights[j] += deltaWeights[j]
            weights += (.1 * (weights[j] - oldWeight))#momentum seems to speed up convergence
        
        for j in range(28):
            if (weights[j] > 50):#this usually prevents overflow of the sigmoid function
                weights[j] = -50#in the rare cases where it doesn't, accuracy drops to zero until random reset occurs
            
            elif (weights[j] < -50):
                weights[j] = -50

        for j in range(7):#updating biases
            biases[j] += deltaBiases[j]
    
    for i in range(30):#validation test
        xList = validationData[i]
        output = forwardPass(xList, weights, biases)
        if (output[0] == validationTeacherNums[i]):
            validationAccuracy += 1.0

    return [weights, biases, trainingAccuracy, validationAccuracy]

def main():
    weightString = ""#initializing variables
    biasString = ""
    f = 0
    f0 = 0
    iris = datasets.load_iris()
    in1 = "A"
    weights = []
    biases = []
    for i in range(28):
        weights.append(0.0)
    
    for i in range(7):
        biases.append(0.0)
    
    wLen = len(weights)
    bLen = len(biases)

    for i in range(150):
        for j in range(4):
            iris.data[i][j] = (iris.data[i][j] / 7.9)#normalizing data to 0-1 range

    while (in1 != "Y" and in1 != "N"):#taking input
        in1 = input("Do you have the files weights.txt and biases.txt? (Y/N)")

        if (in1 == "Y"):#getting weights and biases from file
            f = open("weights.txt", "r+")
            weightString = f.read()
            f0 = open("biases.txt", "r+")
            biasString = f0.read()
        
        elif (in1 == "N"):#creating weights and biases randomly
            f = open("weights.txt", "w+")
            for i in range(28):
                num = random.uniform(-1.0, 1.0)
                numAsStr = str(num) + " "
                weightString += numAsStr
            
            weightString = weightString[1:]
            f.write(weightString)

            f0 = open("biases.txt", "w+")
            for i in range(7):
                num = 0.0
                numAsStr = str(num) + " "
                biasString += numAsStr
            
            biasString = biasString[1:]
        
        else:
            print("Invalid input. Please type Y or N.")
        
    for i in range(wLen):#setting weights from the string
        myStr = ""
        while(weightString != "" and weightString[0] != " "):
            myStr += weightString[0]
            weightString = weightString[1:]

        weightString = weightString[1:]
        myNum = float(myStr)
        weights[i] = myNum
    
    for i in range(bLen):#setting biases from the string
        myStr = ""
        while(biasString != "" and biasString[0] != " "):
            myStr += biasString[0]
            biasString = biasString[1:]
        
        biasString = biasString[1:]
        myNum = float(myStr)
        biases[i] = myNum
    
    in1 = "A"

    while (in1 != "Y" and in1 != "N"):#trains the network or tests against test data
        in1 = input("Would you like to train the network? (Y/N)")

        if (in1 == "Y"):
            in2 = -1
            
            while (in2 < 0):
                in2 = int(input("How many epochs?"))
            
            f.close()#closes files while training so they can be opened in a different mode after
            f0.close()

            for i in range(in2):
                
                temp = backpropagation(iris, weights, biases, .12)#.12 learning rate seems about ideal. From .05 to .25 works moderately well
                #in general, high weights seem to cause overfitting, while low weights seem to cause slow convergence
                
                weights = temp[0]
                biases = temp[1]
                noReset = 0.0

                if (temp[3] > 27):#stops if validation accuracy is high enough, ensuring the model doesn't learn for too long
                    print("Training ended because validation accuracy surpassed 90%")
                    break

                if ((i % 1000) == 999):#every 1000 epochs, outputs the below data, then decides whether to reset
                    print("Epoch: ", end='', sep='')
                    print(i + 1)
                    print("Training accuracy: ", end='', sep='')
                    print(temp[2] * (1 /.9), end='%\n', sep='')
                    print("Validation accuracy: ", end='', sep='')
                    print(temp[3] * (1 / .3), end='%\n', sep='')

                    if (temp[2] < 24 or noReset > 5):#random reset if validation accuracy is less than 80% or 5000 epochs with no reset
                        noReset = 0.0
                        for j in range(28):
                            weights[j] = random.uniform(-1.0, 1.0) 

                        for j in range(7):
                            biases[j] = 0.0

                    else:
                        noReset += 1.0
            
            f = open("weights.txt", "w+")
            f0 = open("biases.txt", "w+")
            weightString = ""
            biasString = ""

            for i in range(wLen):#reforms string from new weights for entry to the text file
                weightString += str(weights[i])
                weightString += " "
            
            weightString = weightString[:-1]
            f.write(weightString)

            for i in range(bLen):#reforms string from new biases for entry to the text file
                biasString += str(biases[i])
                biasString += " "
            
            biasString = biasString[:-1]
            f0.write(biasString)
        
        elif (in1 == "N"):#tests data against test set
            allX = iris.data[:, :4]
            testData = []
            testTeacherNums = []
            testAccuracy = 0.0

            for i in range(150):#determining test set
                if ((i % 5) == 4):
                    testData.append(allX[i])
                    testTeacherNums.append(iris.target[i])
            
            for i in range(30):#testing
                xList = testData[i]
                output = forwardPass(xList, weights, biases)
                if (output[0] == testTeacherNums[i]):
                    testAccuracy += 1.0
            
            print("Test accuracy: ", end='', sep='')
            print(testAccuracy * (1 / .3), end='%\n', sep='')
        
        else:
            print("Invalid input. Please type Y or N.")

    f.close()
    f0.close()

main()
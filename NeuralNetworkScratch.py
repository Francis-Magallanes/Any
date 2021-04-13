import numpy as np
import time
import json

#definition of the activation function and its derivative
def sigmoid(x):
    return 1/(1+np.exp(-x))

#derivative of the sigmoid function
def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

#definition of the cost function
#squared error
def square_error(y_actual, y_expected):
    return (1/2)*np.square(np.subtract(y_actual,y_expected))

#derivative of the squared error
def d_square_error(y_actual, y_expected):
    return np.subtract(y_actual,y_expected)



#definition of the layer class so that the multi-layered neural network can be implemented

#definition of the layer class so that the multi-layered neural network can be implemented

class Layer:
    #this will use a sigmoid activation function
    

    def __init__(self,number_inputs, number_neurons):

        # initializing the weights with random numbers between 0 to 1
        self.weights = np.random.randn(number_neurons, number_inputs) 

        #initialize the bias with one
        self.bias = np.ones((number_neurons,1))

        #default learning rate is 0.5. Don't change these value pls.
        #use the training method to change the learning rate
        self.learning_rate = 0.5 

    def feedforward(self,inputs):
        #apply feedforward algorithm with equation 4 here
        self.previous_output = inputs
        self.Z = np.dot(self.weights,inputs) + self.bias
        self.A = sigmoid(self.Z)
        return self.A
        
    def backpropagate(self, dC_dA):
        #applying the formulas for updatting of the weights and biases
        #refer to equation 8
        dC_dZ = np.multiply(d_sigmoid(self.Z),dC_dA) 

        #normalize the values
        #since the dot product and the np.sum is summed values from the samples.
        # With this code below, the results of the dot product and the summed values
        #will be average, respectively
        dC_dW = 1/ dC_dZ.shape[1] * np.dot(dC_dZ,self.previous_output.T)

        #np.sum is necessary to make the shape of the dC_dB and dC_dZ same
        #np.sum finding the sum along the column matrix to reduce it one column matrix.
        #it sums across the samples
        dC_dB =  1/ dC_dZ.shape[1] * np.sum(dC_dZ, axis=1,keepdims=True)
        #print("dC_dW : {}, dC_dB: {}, dC_dZ: {}". format(dC_dW,dC_dB,dC_dZ)) # uncomment this to see the respectively values

        #update the weights and biases using equation 7
        self.weights = self.weights - np.multiply(self.learning_rate,dC_dW)
        self.bias = self.bias - np.multiply(self.learning_rate,dC_dB)

        #return this value for the backpropagation of the different layer
        return np.dot(self.weights.T,dC_dZ)


#With the layer class, it will provide an abstraction to neural network for organization
class NeuralNetwork:

    # Documentation:
    #num_neurons_per_layer is a list that contains number of neurons per layer
    #each element represent a number of neurons for that layer
    #the minimum number of elements for num_neurons_per_layer is two
    #Ex. 1,2,3 - this signifies that there is one input in the input layer.
    # there are 2 neurons for the first idden layer and also, it has one hidden layer
    #Lastly, there are 3 neurons at the output layer
    def __init__(self,*num_neurons_per_layer):
        #for the checking if num_neurons_per_layer have at least two elements
        assert len(num_neurons_per_layer) >= 2, "num_neurons_per_layer input should have at least two elements"

        #create a list of layers
        self.layers = [ Layer(num_neurons_per_layer[i], num_neurons_per_layer[i+1]) for i in range(len(num_neurons_per_layer) - 1)]
    
    #This constructor will load the saved model based on the filepath_model input
    #filepath_model contains the filepath of the model to be loaded
    #By default, it will check for neural_network_model in the current relative location
    @classmethod
    def load(self,filepath_model="neural_network_model.csv"):
        pass
        

    def train(self, x_train, y_train, learning_rate = 0.5, epochs = 10000):

        if learning_rate != 0.5:
            #change the learning rate accrodingly for each of the layers
            for layer in self.layers:
                layer.learning_rate = learning_rate
        
        #TRAINING SESSION PROPER
        for epoch in range(epochs):

            #feedforward
            input = x_train
            for layer in self.layers:
                input = layer.feedforward(input)
            
            #print("Input: {}".format(input))

            #calculation of the derivative of the cost based on the input variable 
            dCost = d_square_error(input,y_train)

            #uncomment this to know the cost for each of the epoch
            #size_y_train = y_train.shape[0]
            #cost = (1/size_y_train) * square_error(input,y_train).sum()
            #print("Output: {} Cost: {}".format(input,cost)) #uncomment this show the output

            #backpropagation
            for layer in reversed(self.layers):
                dCost = layer.backpropagate(dCost)

    def predict(self,x_predict):
        #implements the feed forward
        y_predict = x_predict
        for layer in self.layers:
                y_predict = layer.feedforward(y_predict)
        
        return y_predict
    
    #filepath input is the filepath were the model to be saved and its name
    #By default, it will save in the current relative location with neural_network_model as its filename 
    #noted that it will use a json file
    def save(self,filepath="neural_network_model.json"):

        #this will create a dictionary for putting the text into the json file
        model ={}

        for i,layer in enumerate(self.layers):

            temp = vars(layer)#this will get the attributes of the layer object

            #this part will only get the important parts of the model
            #which is the bias, weights, and the learning rate
            layer_model =  {}

            layer_model["learning_rate"] = temp["learning_rate"]
            layer_model["weights"] = temp["weights"].tolist()
            layer_model["bias"] = temp["bias"].tolist()

            model[f"layer_{i+1}"] = layer_model
        
        #putting the model into a json file
        with open(filepath,'w') as nn_model:
            json.dump(model,nn_model)

        

    


#With the defined neural network class, instantiated a neural network with input layer (4 inputs),
# 2 hidden layer(with 3  neurons and 2 neurons, respectively), and output layer (with 1 neurons)
neural_network = NeuralNetwork(4,3,2,1)

#TRAINING SESSION
 #possible inputs based on the activity
 #with dimensions of (number of inputs) by (number of training samples) - col x row
 #size: 4 x 8
x_train = np.array([[1,1,0,0,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,1,0,0,1,0,1],[1,0,1,0,0,1,1,0]])

 #possible outputs based on the activity
#with dimensions of (number of outputs) by (number of training samples) - col x row
#size: 1 x 8
y_train = np.array([[1,1,0,0,1,1,0,0]])

start = time.time()
neural_network.train(x_train, y_train, learning_rate = 0.9, epochs = 100000)

#for the predictions
#for the calculating the time of training
time_training = time.time() - start

#input for the neural network
x_test = np.array([[1,1,0,0,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,1,0,0,1,0,1],[1,0,1,0,0,1,1,0]])

y_test = neural_network.predict(x_test)

#show the output
print(y_test)
print("Time training: {}".format(time_training))

#test of the saving of the model
neural_network.save()
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
class Layer:
    #this will use a sigmoid activation function
    
    #this will initialize the layer object 
    #if the layer is already trained, the user can input the bias, weight, and learning rate
    def __init__(self,number_inputs = None, number_neurons = None, bias = None, weights = None, learning_rate=None):

        if(number_inputs is not None and number_neurons is not None):
            # initializing the weights with random numbers between 0 to 1
            self.weights = np.random.randn(number_neurons, number_inputs) 

            #initialize the bias with one
            self.bias = np.ones((number_neurons,1))

            #default learning rate is 0.5. Don't change these value pls.
            #use the training method to change the learning rate
            self.learning_rate = 0.5

        elif (bias is not None and weights is not None and learning_rate is not None):
            self.weights = weights

            self.bias = bias

            self.learning_rate = learning_rate
        
        else:
            raise Exception("The input arguements are not enough to initialize the layer object")



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
    #Ex. [1,2,3] - this signifies that there is one input in the input layer.
    # there are 2 neurons for the first idden layer and also, it has one hidden layer
    #Lastly, there are 3 neurons at the output layer
    #this constructor can load also from the json file which contains the model
    #filepath_model contains the filepath of the model to be loaded
    #By default, it will check for neural_network_model in the current relative location
    def __init__(self,num_neurons_per_layer=[],filepath_model="neural_network_model.json"):

        if(len(num_neurons_per_layer) > 0):
            #for the checking if num_neurons_per_layer have at least two elements
            assert len(num_neurons_per_layer) >= 2, "num_neurons_per_layer input should have at least two elements"

            #create a list of layers
            self.layers = [ Layer(number_inputs = num_neurons_per_layer[i], number_neurons = num_neurons_per_layer[i+1]) for i in range(len(num_neurons_per_layer) - 1)]
        else:
            
            #loading of json file
            with open(filepath_model,  "r") as read_file:
                model = json.load(read_file) #this holds a dictionary within a dictionary
            
            self.layers = []
            for nth_layer in model:
               
                learning_rate = model[nth_layer]["learning_rate"]
                weights = np.array(model[nth_layer]["weights"])
                bias =  np.array(model[nth_layer]["bias"])
                layer = Layer(learning_rate=learning_rate, weights=weights, bias = bias)

                self.layers.append(layer)

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
    def save_model(self,filepath="neural_network_model.json"):

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
            json.dump(model,nn_model, indent= 1)

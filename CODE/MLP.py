import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MLP:

    def __init__(self, inputSize, hiddenSize, outputSize,\
                  learningParameter=0.1, epochs=1000, activationFunc = 'sigmoid'):
        '''
        Initialize the Multi-Layer Perceptron (MLP) neural network
        
        Parameters:
        inputSize: Number of input nodes
        hiddenSize: Number of hidden nodes
        outputSize: Number of output nodes
        learningParameter: Learning rate
        epochs: Number of epochs
        '''
        
        ######### TODO CONSIDER USING BISHOP 1/root(n) instead of -2/hiddenSize, 2/hiddenSize #########
        # self.weightsInputHidden = np.random.uniform(-1/np.sqrt(inputSize), -1/np.sqrt(inputSize), (inputSize, hiddenSize))

        # Initialize weights and biases
        self.weightsInputHidden = np.random.uniform(-2/inputSize, 2/inputSize, (inputSize, hiddenSize))
        self.biasHidden = np.random.rand(1, hiddenSize)
        self.weightsHiddenOutput = np.random.uniform(-2/hiddenSize, 2/hiddenSize, (hiddenSize, outputSize))
        self.biasOutput = np.random.rand(1, outputSize)

        self.prevWeightsInputHidden = np.zeros_like(self.weightsInputHidden)
        self.prevBiasHidden = np.zeros_like(self.biasHidden)
        self.prevWeightsHiddenOutput = np.zeros_like(self.weightsHiddenOutput)
        self.prevBiasOutput = np.zeros_like(self.biasOutput)

        # Initialize learning parameters and epochs
        self.learningParameter = learningParameter
        self.epochs = epochs
        self.activationFunc = activationFunc
    

        self.errors = []


    # GET METHODS
    # To get randomised values for report
    def getWeightsInputHidden(self):
        return self.weightsInputHidden
    
    def getBiasHidden(self):
        return self.biasHidden
    
    def getWeightsHiddenOutput(self):
        return self.weightsHiddenOutput
    
    def getBiasOutput(self):
        return self.biasOutput
    
    #####################################

    # ACTIVATION FUNCTIONS
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoidDerivative(self, x):
        x = np.clip(x, 1e-7, 1 - 1e-7)
        return x * (1 - x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanhDerivative(self, x):
        return 1.0 - x**2
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def reluPrime(self, x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x


    #####################################

    # FORWARD PASS
    def forwardPass(self, X):
        '''
        Forward pass of the neural network

        Parameters:
        X: Input data

        Returns:
        finalOutput: Predicted output
        '''

        self.hiddenInput = np.dot(X, self.weightsInputHidden) + self.biasHidden

        if self.activationFunc == 'sigmoid':
            self.hiddenOutput = self.sigmoid(self.hiddenInput)
        elif self.activationFunc == 'tanh':
            self.hiddenOutput = self.tanh(self.hiddenInput)

        elif self.activationFunc == 'relu':
            self.hiddenOutput = self.relu(self.finalInput)



        self.finalInput = np.dot(self.hiddenOutput, self.weightsHiddenOutput) + self.biasOutput#
        if self.activationFunc == 'sigmoid':
            self.finalOutput = self.sigmoid(self.finalInput)
        elif self.activationFunc == 'tanh':
            self.finalOutput = self.tanh(self.finalInput)
        elif self.activationFunc == 'relu':
            self.finalOutput = self.relu(self.finalInput)

        return self.finalOutput


    # BACKWARD PASS  
    def backwardPass(self, X, y, forwardPassOutput):
        '''
        Backward pass of the neural network
        
        Parameters:
        X: Input data
        y: Actual output
        forwardPassOutput: Predicted output
        
        Returns:
        None
        '''

        # TODO - Use different error calculations (create a separate ErrorCalc class)

        # Ensure y has the correct shape
        # y = y.reshape(-1, 1)

        # Calculate error and delta values
        self.outputError = y - forwardPassOutput

        if self.activationFunc == 'sigmoid':
            self.outputDelta = self.outputError * self.sigmoidDerivative(forwardPassOutput)
        elif self.activationFunc == 'tanh':
            self.outputDelta = self.outputError * self.tanhDerivative(forwardPassOutput)
        elif self.activationFunc == 'relu':
            self.outputDelta = self.outputError * self.reluPrime(forwardPassOutput)

        self.hiddenError = np.dot(self.outputDelta, self.weightsHiddenOutput.T)

        if self.activationFunc == 'sigmoid':
            self.hiddenDelta = self.hiddenError * self.sigmoidDerivative(self.hiddenOutput)
        elif self.activationFunc == 'tanh':
            self.hiddenDelta = self.hiddenError * self.tanhDerivative(self.hiddenOutput)
        elif self.activationFunc == 'relu':
            self.hiddenDelta = self.hiddenError * self.reluPrime(self.hiddenOutput)

        # Update weights and biases

        self.weightsInputHidden += np.dot(X.T, self.hiddenDelta) * self.learningParameter
        self.biasHidden += np.sum(self.hiddenDelta, axis=0) * self.learningParameter

        self.weightsHiddenOutput += np.dot(self.hiddenOutput.T, self.outputDelta) * self.learningParameter
        self.biasOutput += np.sum(self.outputDelta, axis=0) * self.learningParameter



    # TRAINING
    def trainBatch(self, X, y):
        '''
        Train the neural network
        
        Parameters:
        X: Input data
        y: Actual output
        
        Returns:
        None
        '''

        for epoch in range(self.epochs):
            forwardPassOutput = self.forwardPass(X)
            self.backwardPass(X, y, forwardPassOutput)
            error = np.mean(np.square(y - forwardPassOutput))
            self.errors.append(error)

    


    def trainMiniBatch(self, X, y):
        batchSize = 100
        for epoch in range(self.epochs):
            for i in range(0, len(X), batchSize):
                X_batch = X[i:i+batchSize]
                y_batch = y[i:i+batchSize]
                forwardPassOutput = self.forwardPass(X_batch)
                self.backwardPass(X_batch, y_batch, forwardPassOutput)
                
            error = np.mean(np.square(y_batch - forwardPassOutput))
            self.errors.append(error)

    


    def trainSeq(self, X, y):

        for epoch in range(self.epochs):

            for i in range(len(X)):
                predictors = X[i].reshape(1, -1)
                target = y[i].reshape(1, -1)

                forwardPassOutput = self.forwardPass(predictors)

                self.backwardPass(predictors, target, forwardPassOutput)

            
            error = np.mean(np.square(target - forwardPassOutput))

            self.errors.append(error)

        




# Example usage with Pandas DataFrame

if __name__ == "__main__":

    #Create a pandas dataframe
    data = {
        'feature1': [0, 0, 1, 1],
        'feature2': [0, 1, 0, 1],
        'label': [0, 1, 1, 0]
    }

    df = pd.DataFrame(data)

    print(df)

    # Prepare input (X) and output (y)
    X = df[['feature1', 'feature2']].values
    y = df['label'].values.reshape(-1, 1)


    # Scale Units using minmax scaling
    X = (X - X.min()) / (X.max() - X.min())
    y = (y - y.min()) / (y.max() - y.min())

    mlp = MLP(2, 2, 1)

    mlp.train(X, y)

    print("Final weights between input and hidden layer: ")
    print(mlp.getWeightsInputHidden())
    print("Final bias for hidden layer: ")
    print(mlp.getBiasHidden())
    print("Final weights between hidden and output layer: ")
    print(mlp.getWeightsHiddenOutput())
    print("Final bias for output layer: ")
    print(mlp.getBiasOutput())

    mlp.forwardPass(X)
    
    # Undo Scaling
    X = X * (X.max() - X.min()) + X.min()
    y = y * (y.max() - y.min()) + y.min()



    print("Predictions: ")
    print(mlp.forwardPass(X))

    print("Actual: ")
    print(y)
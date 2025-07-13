from MLP import MLP
import numpy as np



class Momentum(MLP):

    def __init__(self, inputSize, hiddenSize, outputSize, learningParameter=0.1, epochs=1000, activationFunc='sigmoid', alpha = 0.9):
        super().__init__(inputSize, hiddenSize, outputSize, learningParameter, epochs, activationFunc)
        self.alpha = alpha


    def backwardPass(self, X, y, forwardPassOutput):
        '''
        Backward pass of the neural network using momentum
        
        Parameters:
        X: Input data
        y: Actual output
        forwardPassOutput: Predicted output
        
        Returns:
        None
        '''

        # Ensure y has the correct shape
        y = y.reshape(-1, 1)

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


        momentumInputHidden = self.alpha * self.prevWeightsInputHidden
        momentumBiasHidden = self.alpha * self.prevBiasHidden
        momentumHiddenOutput = self.alpha * self.prevWeightsHiddenOutput
        momentumBiasOutput = self.alpha * self.prevBiasOutput

        weightUpdateInputHidden = np.dot(X.T, self.hiddenDelta) * self.learningParameter
        self.weightsInputHidden += momentumInputHidden + weightUpdateInputHidden
        self.prevWeightsInputHidden = weightUpdateInputHidden


        biasUpdateHidden = np.sum(self.hiddenDelta, axis=0) * self.learningParameter
        self.biasHidden += momentumBiasHidden + biasUpdateHidden
        self.prevBiasHidden = biasUpdateHidden

        weightUpdateHiddenOutput = np.dot(self.hiddenOutput.T, self.outputDelta) * self.learningParameter
        self.weightsHiddenOutput += np.clip(momentumHiddenOutput + weightUpdateHiddenOutput, -1e10, 1e10)
        self.prevWeightsHiddenOutput = weightUpdateHiddenOutput

        biasUpdateOutput = np.sum(self.outputDelta, axis=0) * self.learningParameter
        self.biasOutput += np.clip(momentumBiasOutput + biasUpdateOutput, -1e10, 1e10)
        self.prevBiasOutput = biasUpdateOutput



# class BoldDriver(MLP):

#     def __init__(self, inputSize, hiddenSize, outputSize, learningParameter=0.1, epochs=1000, activationFunc='sigmoid', increaseFactor=1.04, decreaseFactor=0.7):
#         super().__init__(inputSize, hiddenSize, outputSize, learningParameter, epochs, activationFunc)
#         self.increaseFactor = increaseFactor 
#         self.decreaseFactor = decreaseFactor
#         self.previousError = float('inf')




#     # Implement Bold Driver improvement
#     def trainBatch(self, X, y):
#         '''
#         Train the neural network
        
#         Parameters:
#         X: Input data
#         y: Actual output
        
#         Returns:
#         None
#         '''

#         for epoch in range(self.epochs):
#             forwardPassOutput = self.forwardPass(X)
            
#             error = np.mean(np.square(y - forwardPassOutput))

#             self.backwardPass(X, y, forwardPassOutput)


#             # FIXME instead set new learnin

#             if error < self.previousError: # Change to self.previousError - error < 4%
#                 self.learningParameter *= self.increaseFactor

#             else:
#                 self.learningParameter *= self.decreaseFactor
#                 self.weightsInputHidden -= self.prevWeightsInputHidden
#                 self.biasHidden -= self.prevBiasHidden
#                 self.weightsHiddenOutput -= self.prevWeightsHiddenOutput
#                 self.biasOutput -= self.prevBiasOutput

            

#             self.prevWeightsInputHidden = np.dot(X.T, self.hiddenDelta) * self.learningParameter
#             self.prevBiasHidden = np.sum(self.hiddenDelta, axis=0) * self.learningParameter
#             self.prevWeightsHiddenOutput = np.dot(self.hiddenOutput.T, self.outputDelta) * self.learningParameter
#             self.prevBiasOutput = np.sum(self.outputDelta, axis=0) * self.learningParameter


            
#             self.previousError = error
#             self.errors.append(error)





#     def trainMiniBatch(self, X, y):
#         batchSize = 100
#         for epoch in range(self.epochs):
#             for i in range(0, len(X), batchSize):
#                 X_batch = X[i:i+batchSize]
#                 y_batch = y[i:i+batchSize]#

#                 forwardPassOutput = self.forwardPass(X_batch)

#                 self.backwardPass(X_batch, y_batch, forwardPassOutput)

#                 error = np.mean(np.square(y_batch - forwardPassOutput))

#                 if error < self.previousError: # Change to self.previousError - error < 4%
#                     self.learningParameter *= self.increaseFactor

#                 else:
#                     self.learningParameter *= self.decreaseFactor
#                     self.weightsInputHidden -= self.prevWeightsInputHidden
#                     self.biasHidden -= self.prevBiasHidden
#                     self.weightsHiddenOutput -= self.prevWeightsHiddenOutput
#                     self.biasOutput -= self.prevBiasOutput

            

#                 self.prevWeightsInputHidden = np.dot(X_batch.T, self.hiddenDelta) * self.learningParameter
#                 self.prevBiasHidden = np.sum(self.hiddenDelta, axis=0) * self.learningParameter
#                 self.prevWeightsHiddenOutput = np.dot(self.hiddenOutput.T, self.outputDelta) * self.learningParameter
#                 self.prevBiasOutput = np.sum(self.outputDelta, axis=0) * self.learningParameter


                
                
                
#             self.errors.append(error)




#     def trainSeq(self, X, y):

#         for epoch in range(self.epochs):

#             for i in range(len(X)):
#                 predictors = X[i].reshape(1, -1)
#                 target = y[i].reshape(1, -1)

#                 forwardPassOutput = self.forwardPass(predictors)

#                 error = np.mean(np.square(target - forwardPassOutput))

#                 self.backwardPass(predictors, target, forwardPassOutput)

                
#                 if error < self.previousError: # Change to self.previousError - error < 4%
#                     self.learningParameter *= self.increaseFactor

#                 else:
#                     self.learningParameter *= self.decreaseFactor
#                     self.weightsInputHidden -= self.prevWeightsInputHidden
#                     self.biasHidden -= self.prevBiasHidden
#                     self.weightsHiddenOutput -= self.prevWeightsHiddenOutput
#                     self.biasOutput -= self.prevBiasOutput

            

#                 self.prevWeightsInputHidden = np.dot(predictors.T, self.hiddenDelta) * self.learningParameter
#                 self.prevBiasHidden = np.sum(self.hiddenDelta, axis=0) * self.learningParameter
#                 self.prevWeightsHiddenOutput = np.dot(self.hiddenOutput.T, self.outputDelta) * self.learningParameter
#                 self.prevBiasOutput = np.sum(self.outputDelta, axis=0) * self.learningParameter


#             self.errors.append(error)





class WeightDecay(MLP):

    def __init__(self, inputSize, hiddenSize, outputSize, learningParameter=0.1, epochs=1000, activationFunc='sigmoid'):
        super().__init__(inputSize, hiddenSize, outputSize, learningParameter, epochs, activationFunc)


    def backwardPass(self, X, y, forwardPassOutput):

        self.outputError = y - forwardPassOutput + self.decay

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

        self.hiddenDelta = np.clip(self.hiddenDelta, -1, 1)
        self.outputDelta = np.clip(self.outputDelta, -1, 1)

        # Update weights and biases

        self.weightsInputHidden += np.dot(X.T, self.hiddenDelta) * self.learningParameter
        self.biasHidden += np.sum(self.hiddenDelta, axis=0) * self.learningParameter

        self.weightsHiddenOutput += np.dot(self.hiddenOutput.T, self.outputDelta) * self.learningParameter
        self.biasOutput += np.sum(self.outputDelta, axis=0) * self.learningParameter

    


    def trainBatch(self, X, y):
        '''
        Train the neural network with weight decay using batch.

        Parameters:
        X: Input data
        y: Actual output

        Returns:
        None
        '''
        for epoch in range(self.epochs):
            forwardPassOutput = self.forwardPass(X)

            if epoch == 0:
                self.decay = 0

            else:
            
                # Decay 𝛽Ω
                self.decay = (1/epoch) * (1 / (2 * X.shape[0])) * (
                np.sum(np.square(self.weightsInputHidden)) + 
                np.sum(np.square(self.weightsHiddenOutput))
                )

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

                if epoch == 0:
                    self.decay = 0

                else:
                
                    # Decay 𝛽Ω
                    self.decay = (1/epoch) * (1 / (2 * X_batch.shape[0])) * (
                    np.sum(np.square(self.weightsInputHidden)) + 
                    np.sum(np.square(self.weightsHiddenOutput))
                    )

                self.backwardPass(X_batch, y_batch, forwardPassOutput)
                
                error = np.mean(np.square(y_batch - forwardPassOutput))
            
            self.errors.append(error)



    def trainSeq(self, X, y):

        for epoch in range(self.epochs):

            for i in range(len(X)):
                predictors = X[i].reshape(1, -1)
                target = y[i].reshape(1, -1)

                forwardPassOutput = self.forwardPass(predictors)

                if epoch == 0:
                    self.decay = 0

                else:
                
                    # Decay 𝛽Ω
                    self.decay = (1/epoch) * (1 / (2 * X.shape[0])) * (
                    np.sum(np.square(self.weightsInputHidden)) + 
                    np.sum(np.square(self.weightsHiddenOutput))
                    )

                error = np.mean(np.square(target - forwardPassOutput))

                self.backwardPass(predictors, target, forwardPassOutput)


            self.errors.append(error)



class Annealing(MLP):

    def __init__(self, inputSize, hiddenSize, outputSize, learningParameter=0.1, epochs=1000, activationFunc='sigmoid', p=0.1, q=0.01, r=0.5):
        super().__init__(inputSize, hiddenSize, outputSize, learningParameter, epochs, activationFunc)

        self.p = p  # Final learning rate
        self.q = q  # Initial learning rate
        self.r = r  # Maximum epochs

    def annealing_function(self, epoch):
        '''
        Compute the annealed parameter using the sigmoid-based formula.

        Parameters:
        epoch: Current epoch (integer)

        Returns:
        learningRate: Adjusted learning rate for the current epoch
        '''
        x = epoch
        learningRate = self.p + (self.q - self.p) * (1 - (1 / (1 + np.exp((10 - 20 * x) / self.r))))
        return learningRate
    


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

            self.learningParameter = self.annealing_function(epoch)

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

                self.learningParameter = self.annealing_function(epoch)

                forwardPassOutput = self.forwardPass(X_batch)
                self.backwardPass(X_batch, y_batch, forwardPassOutput)
                
                error = np.mean(np.square(y_batch - forwardPassOutput))


            self.errors.append(error)



    def trainSeq(self, X, y):

        for epoch in range(self.epochs):

            for i in range(len(X)):
                predictors = X[i].reshape(1, -1)
                target = y[i].reshape(1, -1)

                self.learningParameter = self.annealing_function(epoch)

                forwardPassOutput = self.forwardPass(predictors)
                
                error = np.mean(np.square(target - forwardPassOutput))

                self.backwardPass(predictors, target, forwardPassOutput)

            

            self.errors.append(error)


class AllImprovements(MLP):

    def __init__(self, inputSize, hiddenSize, outputSize, learningParameter=0.1, epochs=1000, activationFunc='sigmoid', alpha = 0.9, p=0.1, q=0.01, r=0.5):
        super().__init__(inputSize, hiddenSize, outputSize, learningParameter, epochs, activationFunc)

        self.alpha = alpha

        self.p = p  # Final learning rate
        self.q = q  # Initial learning rate
        self.r = r  # Maximum epochs

    def annealing_function(self, epoch):
        '''
        Compute the annealed parameter using the sigmoid-based formula.

        Parameters:
        epoch: Current epoch (integer)

        Returns:
        learningRate: Adjusted learning rate for the current epoch
        '''
        x = epoch
        learningRate = self.p + (self.q - self.p) * (1 - (1 / (1 + np.exp((10 - 20 * x) / self.r))))
        return learningRate

    
    def backwardPass(self, X, y, forwardPassOutput):

        self.outputError = y - forwardPassOutput + self.decay

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

        self.hiddenDelta = np.clip(self.hiddenDelta, -1, 1)
        self.outputDelta = np.clip(self.outputDelta, -1, 1)

        momentumInputHidden = self.alpha * self.prevWeightsInputHidden
        momentumBiasHidden = self.alpha * self.prevBiasHidden
        momentumHiddenOutput = self.alpha * self.prevWeightsHiddenOutput
        momentumBiasOutput = self.alpha * self.prevBiasOutput

        weightUpdateInputHidden = np.dot(X.T, self.hiddenDelta) * self.learningParameter
        self.weightsInputHidden += momentumInputHidden + weightUpdateInputHidden
        self.prevWeightsInputHidden = weightUpdateInputHidden


        biasUpdateHidden = np.sum(self.hiddenDelta, axis=0) * self.learningParameter
        self.biasHidden += momentumBiasHidden + biasUpdateHidden
        self.prevBiasHidden = biasUpdateHidden

        weightUpdateHiddenOutput = np.dot(self.hiddenOutput.T, self.outputDelta) * self.learningParameter
        self.weightsHiddenOutput += np.clip(momentumHiddenOutput + weightUpdateHiddenOutput, -1e10, 1e10)
        self.prevWeightsHiddenOutput = weightUpdateHiddenOutput

        biasUpdateOutput = np.sum(self.outputDelta, axis=0) * self.learningParameter
        self.biasOutput += np.clip(momentumBiasOutput + biasUpdateOutput, -1e10, 1e10)
        self.prevBiasOutput = biasUpdateOutput



    def trainBatch(self, X, y):
        for epoch in range(self.epochs):

            self.learningParameter = self.annealing_function(epoch)

            forwardPassOutput = self.forwardPass(X)

            if epoch == 0:
                self.decay = 0

            else:
            
                # Decay 𝛽Ω
                self.decay = (1/epoch) * (1 / (2 * X.shape[0])) * (
                np.sum(np.square(self.weightsInputHidden)) + 
                np.sum(np.square(self.weightsHiddenOutput))
                )

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

                if epoch == 0:
                    self.decay = 0

                else:
                
                    # Decay 𝛽Ω
                    self.decay = (1/epoch) * (1 / (2 * X_batch.shape[0])) * (
                    np.sum(np.square(self.weightsInputHidden)) + 
                    np.sum(np.square(self.weightsHiddenOutput))
                    )

                self.backwardPass(X_batch, y_batch, forwardPassOutput)
                
                error = np.mean(np.square(y_batch - forwardPassOutput))
            
            self.errors.append(error)



    def trainSeq(self, X, y):

        for epoch in range(self.epochs):

            for i in range(len(X)):
                predictors = X[i].reshape(1, -1)
                target = y[i].reshape(1, -1)

                forwardPassOutput = self.forwardPass(predictors)

                if epoch == 0:
                    self.decay = 0

                else:
                
                    # Decay 𝛽Ω
                    self.decay = (1/epoch) * (1 / (2 * X.shape[0])) * (
                    np.sum(np.square(self.weightsInputHidden)) + 
                    np.sum(np.square(self.weightsHiddenOutput))
                    )

                error = np.mean(np.square(target - forwardPassOutput))

                self.backwardPass(predictors, target, forwardPassOutput)


            self.errors.append(error)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from MLP import MLP
from improvements import *

from scipy.stats import pearsonr



'''
DATA PREPROCESSING
'''

df = pd.read_excel('Ouse93-96 - Student.xlsx', usecols=range(0, 9))


# Change value at [0, 0] to 'Date'
df.iloc[0, 0] = "Date"


# Make the first row the column names

df.columns = df.iloc[0]
df = df.drop(df.index[0]) 

################## TODO REMOVE ##################

# df.rename(columns=df.iloc[0], inplace = True)

# df.drop(df.index[0], inplace = True)

# Test values to see changes
# 99, 119 --> subract 2
# print(df.loc[117])

#################################################


# Change all values that are non numeric, ie. "a", "#", to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Set all values that are -999 to NaN
df = df.replace(-999, np.nan)

df['Date'] = pd.to_datetime(df['Date'])





# Set any data that is 4 standard deviations away from the mean to NaN from df 
# Set any data that is 1.5 * IQR away from the mean to NaN from df

def removeOutliers(df, column):
    '''
    Remove outliers from a column in a dataframe

    Args:
        df: pandas dataframe
        column: string, column name

    Returns:
        df: pandas dataframe with outliers removed
    '''
    std = df[column].std()
    mean = df[column].mean()
    std4 = df[column].std() * 4
    for i in range(1, len(df.index)):
        if (df.loc[i, column] > mean + std4) or \
            (df.loc[i, column] < mean - std4):
            df.loc[i, column] = np.nan

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    for i in range(1, len(df.index)):
        if (df.loc[i, column] > q3 + 1.5 * iqr) or \
              (df.loc[i, column] < q1 - 1.5 * iqr):
            df.loc[i, column] = np.nan

    return df
      


df = removeOutliers(df, "Skelton")
df = removeOutliers(df, "Crakehill")
df = removeOutliers(df, "Skip Bridge")
df = removeOutliers(df, "Westwick")
df = removeOutliers(df, "Arkengarthdale")
df = removeOutliers(df, "East Cowton")
df = removeOutliers(df, "Malham Tarn")
df = removeOutliers(df, "Snaizeholme")



###################### CREATE COLUMNS FOR MOVING AVERAGES & MAKE THEM PREDICTORS ######################


# Calculate the moving averages for skelton
# df["Skelton 7 day moving average"] = df["Skelton"].rolling(window=7).mean()
# df["Skelton 30 day moving average"] = df["Skelton"].rolling(window=30).mean()
# df["Skelton 60 day moving average"] = df["Skelton"].rolling(window=60).mean()


# Calculate the moving averages for all the other columns in one go
# for column in df.columns[1:]:
    # df[column + " 7 day moving average"] = df[column].rolling(window=7).mean()
    # df[column + " 30 day moving average"] = df[column].rolling(window=30).mean()
    # df[column + " 60 day moving average"] = df[column].rolling(window=60).mean()

########################################################################################################


# Test by lagging skelton by 1 day
df["Skelton lag 1"] = df["Skelton"].shift(1)
df["Skelton Tomorrow"] = df["Skelton"].shift(-1)



# Interpolate values for all the NaN values in df
df = df.interpolate()

# Drop first row due to skelton lag 1 being NaN
df = df.drop(df.index[0])



# Plot a heat map of the correlation matrix excluding the date column and use coolwarm colour map
df_corr = df.drop("Date", axis=1)

'''
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df_corr.corr(), cmap='coolwarm', annot=True, fmt=".2f")
sns.set_context("paper", rc={"figure.figsize": (8, 6)})
plt.title("Correlation Matrix")
plt.xticks(rotation=30)
plt.show()
'''


'''
DATA PREPARATION
'''

# Randomise data
# df = df.sample(n=len(df))


# Split data into predictors and predictand


predictors = df.drop(["Date", "Skelton"], axis=1)
predictand = df["Skelton"]

# print(predictors)
# print(predictand)



# Normalise the data

# predictorsMean = predictors.mean()
# predictorsStd = predictors.std()
# predictandMean = predictand.mean()
# predictandStd = predictand.std()

# predictors = (predictors - predictors.mean()) / predictors.std()
# predictand = (predictand - predictand.mean()) / predictand.std()


# Scale the data using max-min scaling

predictorsMax = predictors.max()
predictorsMin = predictors.min()

predictandMax = predictand.max()
predictandMin = predictand.min()

predictors = 0.8 * (predictors - predictorsMin) / (predictorsMax - predictorsMin) + 0.1
predictand = 0.8 * (predictand - predictandMin) / (predictandMax - predictandMin) + 0.1



# Split the data into training, testing and validation data
# 50% training, 25% testing, 25% validation

trainSize = int(0.5 * len(df.index))
testSize = int(0.25 * len(df.index))
validationSize = int(0.25 * len(df.index))

trainPredictors = predictors[:trainSize]
trainPredictand = predictand[:trainSize]

testPredictors = predictors[trainSize:trainSize + testSize]
testPredictand = predictand[trainSize:trainSize + testSize]

validationPredictors = predictors[trainSize + testSize:]
validationPredictand = predictand[trainSize + testSize:]


'''
APPLYING MLP TO DATA
'''

epochs = 1000
learningParameter = 0.1



mlp = MLP(predictors.shape[1], 8, 1, epochs=epochs, learningParameter=learningParameter)#, activationFunc='tanh')

mlpMomentum = Momentum(predictors.shape[1], 8, 1, epochs=epochs, learningParameter=learningParameter, alpha=0.9)#, activationFunc='tanh')
# mlpBoldDriver = BoldDriver(predictors.shape[1], 8, 1, epochs=epochs, learningParameter=learningParameter)#, activationFunc='tanh')
mlpWeightDecay = WeightDecay(predictors.shape[1], 8, 1, epochs=epochs, learningParameter=learningParameter)#, activationFunc='tanh')
mlpAnnealing = Annealing(predictors.shape[1], 8, 1, epochs=epochs, learningParameter=learningParameter)#, activationFunc='tanh')
mlpAllImprovements = AllImprovements(predictors.shape[1], 8, 1, epochs=epochs, learningParameter=learningParameter, r=300)


X = trainPredictors.values
y = trainPredictand.values.reshape(-1, 1)





mlp.trainSeq(X, y)
mlpWeightDecay.trainSeq(X, y)

mlpMomentum.trainBatch(X, y)
# mlpBoldDriver.trainMiniBatch(X, y)

mlpAnnealing.trainBatch(X, y)
mlpAllImprovements.trainBatch(X, y)



print(mlp.errors[-1])

print("Momentum: ", mlpMomentum.errors[-1])
# print("Bold Driver: ", mlpBoldDriver.errors[-1])
print("Weight Decay: ", mlpWeightDecay.errors[-1])
print("Annealing", mlpAnnealing.errors[-1])

print("All: ", mlpAllImprovements.errors[-1])




predictions = mlp.forwardPass(validationPredictors)


predictionsMomentum = mlpMomentum.forwardPass(testPredictors)
# predictionsBD = mlpBoldDriver.forwardPass(testPredictors)
predictionsWD = mlpWeightDecay.forwardPass(validationPredictors)
predictionsAnnealing = mlpAnnealing.forwardPass(testPredictors)

predictionsAll = mlpAllImprovements.forwardPass(testPredictors)





# Plot error against epochs

figErrors, axErrors = plt.subplots()
figErrors.set_figheight(5)
figErrors.set_figwidth(15)
axErrors.plot(range(len(mlp.errors)), mlp.errors) 
axErrors.set_xlabel("Epochs")
axErrors.set_ylabel("Error")
axErrors.set_title("Error vs Epochs without improvements")

'''

figErrorsMomentum, axErrorsMomentum = plt.subplots()
figErrorsMomentum.set_figheight(5)
figErrorsMomentum.set_figwidth(15)
axErrorsMomentum.plot(range(len(mlpMomentum.errors)), mlpMomentum.errors)
axErrorsMomentum.set_xlabel("Epochs")
axErrorsMomentum.set_ylabel("Error")
axErrorsMomentum.set_title("Errors vs Epochs with Momentum")


# figErrorsBD, axErrorsBD = plt.subplots()
# figErrorsBD.set_figheight(5)
# figErrorsBD.set_figwidth(15)
# axErrorsBD.plot(range(len(mlpBoldDriver.errors)), mlpBoldDriver.errors)
# axErrorsBD.set_title("Errors vs Epochs with Bold Driver")

'''

figErrorsWD, axErrorsWD = plt.subplots()
figErrorsWD.set_figheight(5)
figErrorsWD.set_figwidth(15)
axErrorsWD.plot(range(len(mlpWeightDecay.errors)), mlpWeightDecay.errors)
axErrorsWD.set_xlabel("Epochs")
axErrorsWD.set_ylabel("Error")
axErrorsWD.set_title("Errors vs Epochs with Weight Decay")
'''

figErrorsAnnealing, axErrorsAnnealing = plt.subplots()
figErrorsAnnealing.set_figheight(5)
figErrorsAnnealing.set_figwidth(15)
axErrorsAnnealing.plot(range(len(mlpAnnealing.errors)), mlpAnnealing.errors)
axErrorsAnnealing.set_xlabel("Epochs")
axErrorsAnnealing.set_ylabel("Error")
axErrorsAnnealing.set_title("Errors vs Epochs with Annealing")



figErrorsAll, axErrorsAll = plt.subplots()
figErrorsAll.set_figheight(5)
figErrorsAll.set_figwidth(15)
axErrorsAll.plot(range(len(mlpAllImprovements.errors)), mlpAllImprovements.errors)
axErrorsAll.set_xlabel("Epochs")
axErrorsAll.set_ylabel("Error")
axErrorsAll.set_title("Errors vs Epochs with All")

'''

plt.show()



# Undo normalisation

# predictions = predictions * predictandStd + predictandMean
# predictionsMomentum = predictionsMomentum * predictandStd + predictandMean
# predictionsBD = predictionsBD * predictandStd + predictandMean
# predictionsWD = predictionsWD * predictandStd + predictandMean
# predictionsAnnealing = predictionsAnnealing * predictandStd + predictandMean

# testPredictand = testPredictand * predictandMean + predictandMean

# Undo minmax scaling

predictions = (predictions - 0.1) * (predictandMax - predictandMin) + predictandMin

predictionsMomentum = (predictionsMomentum - 0.1) * (predictandMax - predictandMin) + predictandMin
# predictionsBD = (predictionsBD - 0.1) * (predictandMax - predictandMin) + predictandMin
predictionsWD = (predictionsWD - 0.1) * (predictandMax - predictandMin) + predictandMin
predictionsAnnealing = (predictionsAnnealing - 0.1) * (predictandMax - predictandMin) + predictandMin

predictionsAll = (predictionsAll - 0.1) * (predictandMax - predictandMin) + predictandMin

testPredictand = (testPredictand - 0.1) * (predictandMax - predictandMin) + predictandMin




figWithoutImprovements, axWithoutImprovements = plt.subplots()
figWithoutImprovements.set_figheight(5)
figWithoutImprovements.set_figwidth(15)
axWithoutImprovements.plot(range(len(predictions)), predictions, label="Predictions")
axWithoutImprovements.plot(range(len(testPredictand)), testPredictand, label="Actual")
axWithoutImprovements.set_xlabel("Samples")
axWithoutImprovements.set_ylabel("Mean Daily Flow")
axWithoutImprovements.set_title("Predictions vs Actual")
axWithoutImprovements.legend()

'''

figWithMomentum, axWithMomentum = plt.subplots()
figWithMomentum.set_figheight(5)
figWithMomentum.set_figwidth(15)
axWithMomentum.plot(range(len(predictions)), predictionsMomentum, label="Predictions With Momentum")
axWithMomentum.plot(range(len(testPredictand)), testPredictand, label="Actual")
axWithMomentum.set_xlabel("Samples")
axWithMomentum.set_ylabel("Mean Daily Flow")
axWithMomentum.set_title("Predictions vs Actual")
axWithMomentum.legend()

# figWithBD, axWithBD = plt.subplots()
# figWithBD.set_figheight(5)
# figWithBD.set_figwidth(15)
# axWithBD.plot(range(len(predictionsBD)), predictionsBD, label="Predictions With Bold Driver")
# axWithBD.plot(range(len(testPredictand)), testPredictand, label="Actual")
# axWithBD.set_title("Predictions vs Actual")
# axWithBD.legend()

'''
figWithWD, axWithWD = plt.subplots()
figWithWD.set_figheight(5)
figWithWD.set_figwidth(15)
axWithWD.plot(range(len(predictionsWD)), predictionsWD, label="Predictions With Weight Decay")
axWithWD.plot(range(len(testPredictand)), testPredictand, label="Actual")
axWithWD.set_xlabel("Samples")
axWithWD.set_ylabel("Mean Daily Flow")
axWithWD.set_title("Predictions vs Actual")
axWithWD.legend()
'''
figWithAnnealing, axWithAnnealing = plt.subplots()
figWithAnnealing.set_figheight(5)
figWithAnnealing.set_figwidth(15)
axWithAnnealing.plot(range(len(predictionsAnnealing)), predictionsAnnealing, label="Predictions With Annealing")
axWithAnnealing.plot(range(len(testPredictand)), testPredictand, label="Actual")
axWithAnnealing.set_xlabel("Samples")
axWithAnnealing.set_ylabel("Mean Daily Flow")
axWithAnnealing.set_title("Predictions vs Actual")
axWithAnnealing.legend()


figWithAll, axWithAll = plt.subplots()
figWithAll.set_figheight(5)
figWithAll.set_figwidth(15)
axWithAll.plot(range(len(predictionsAnnealing)), predictionsAnnealing, label="Predictions With All")
axWithAll.plot(range(len(testPredictand)), testPredictand, label="Actual")
axWithAll.set_xlabel("Samples")
axWithAll.set_ylabel("Mean Daily Flow")
axWithAll.set_title("Predictions vs Actual")
axWithAll.legend()

'''

plt.show()






actual = validationPredictand.ravel()
predicted = predictionsWD.ravel()

# Undo normalization for actual and predicted values (if needed)
# actual = (actual - 0.1) * (predictandMax - predictandMin) + predictandMin
# predicted = (predicted - 0.1) * (predictandMax - predictandMin) + predictandMin

# Calculate Pearson correlation coefficient
correlation, p_value = pearsonr(actual, predicted)
print(f"Pearson Correlation Coefficient: {correlation:.4f}, p-value: {p_value:.4e}")

# Fit a linear regression line
slope, intercept = np.polyfit(actual, predicted, 1)
regression_line = np.poly1d([slope, intercept])

# Plot the scatter plot and regression line
plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, label="Data Points", color="blue", alpha=0.6)
plt.plot(actual, regression_line(actual), label=f"Regression Line (y={slope:.2f}x+{intercept:.2f})", color="red")

# Add correlation coefficient to the graph
plt.text(
    0.05, 0.95,  # Position (adjust as needed)
    f"Pearson r = {correlation:.4f}",
    fontsize=12,
    color="black",
    transform=plt.gca().transAxes,  # Use axes coordinates (0, 0 is bottom-left, 1, 1 is top-right)
)

# Add labels, title, and legend
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Predictions vs Actual")
plt.legend()
plt.grid(True)
plt.show()



# predictionsValidations = mlp.forwardPass(validationPredictors)

# predictionsValidations = (predictionsValidations - 0.1) * (predictandMax - predictandMin) + predictandMin
# validationPredictand = (validationPredictand - 0.1) * (predictandMax - predictandMin) + predictandMin


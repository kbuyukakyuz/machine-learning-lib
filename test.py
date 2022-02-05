import regression as rg
import pandas as pd
import numpy as np

# Load the data from regression module
data = rg.load("Walmart_Store_sales")

# Date has to be deleted in order to use dot product
data = data.drop(["Date"], axis=1)

data = data.values

# Turn data into numpy arrays, separate features and target
# Reshape the array
x,y = rg.process(data)
# Normalize the array
x,y = rg.normalize(x, y)

#Get the train and test data for a given size, randon seed state also can be changed
x_train, x_test, y_train, y_test = rg.dissamble_data(x, y, sizeof_test=1/4)
#Initialize the model for eta and number of iterations, for normalized data less conservative eta can be used 
grad, cost_list = rg.model(x, y, 0.5, 10000)
#Graph shows how cost function converges to a value
#Also confusion matrix can be used from regression module
print(rg.accuracy(x_test, y_test, grad))
#91% Accuracy for chosen values
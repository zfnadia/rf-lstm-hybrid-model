import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from sklearn import metrics

rcParams['figure.figsize'] = 22, 10

df = pd.read_csv('../csv_files/main_dataset_2.csv')
print(df.info())

timestamp = np.array(df['timestamp'])

df = df.drop(['timestamp'], axis=1)

# Labels are the values we want to predict
labels = np.array(df['total_energy'])

# Remove the labels from the features
# axis 1 refers to the columns
df = df.drop('total_energy', axis=1)

# Saving feature names for later use
feature_list = list(df.columns)
# Convert to numpy array
df = np.array(df)

# Split the data into training and testing sets
# test_size=0.3 means 70% data is used to train the model and the rest is used for test
# The random state to 42 means the results will be the same each time I run the split for reproducible results
x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.1, random_state=42)
print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

#Setting up the model. And that’s pretty much how we create a linear regression model using SciKit.
model = LinearRegression(fit_intercept=True, normalize=False, n_jobs=None)

#we have to fit this model to our data, in other words
#we have to make it “learn” using our training data. For that, its just one other line of code

model.fit(x_train, y_train)

#Now, my model is trained with the training set I created. We can now start testing the model with the testing dataset we have.
# For that, we add one more line to my code

y_pred = model.predict(x_test)

#The next step is to see how well our prediction is working. For this, we’ll use the MatPlotLib library.
#First, we’ll plot the actual values from our dataset against the predicted values for the training set.
#This will tell us how accurate our model is. After that, we’ll make another plot with the test set.
#In both cases, we’ll be using a scatter plot. We’ll plot the actual values (from the dataset) in red, and our model’s predictions in blue.
#This way, we’ll be able to easily differentiate the two.
# plot.scatter(x_train, y_train, color = 'red')
# plot.plot(x_train, model.predict(x_train), color = 'blue')
# plot.title('Closing price vs Opening price (Training set)')
# plot.xlabel('Opening Price')
# plot.ylabel('Closing Price')
# plot.show()
#
# plot.scatter(x_test, y_test, color = 'red')
# plot.plot(x_test, model.predict(x_test), color = 'blue')
# plot.title('Closing Price vs Opening Price (Test set)')
# plot.xlabel('Opening Price')
# plot.ylabel('Closing Price')
# plot.show()

plt.plot(y_test, marker='.', label="true", linewidth=3, markersize='12')
plt.plot(y_pred, 'r', marker='.', label="prediction", linewidth=1, markersize='12')
plt.ylabel('electricity Consumption (MKWh)')
plt.xlabel('time step')
plt.legend()
# plt.savefig('../assets_lstm_2/total_test_vs_train_all_feat.png', bbox_inches='tight')
plt.savefig('../assets_lstm_2/total_test_vs_train_imp_feat.png', bbox_inches='tight')
plt.show()

# The mean squared error
print("Mean squared error: %.2f" % np.mean((model.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(x_test, y_test))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('MAPE: ', metrics.mean_absolute_error(y_test, y_pred, multioutput='uniform_average'))

accuracy = model.score(x_test,y_test)
print(accuracy*100,'%')



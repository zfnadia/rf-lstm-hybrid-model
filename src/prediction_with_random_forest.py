import datetime
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

df = pd.read_csv('../csv_files/main_dataset.csv')
print(df.info())
# df["max_temp"] = df.max_temp.astype(float)
# df["total_gas"] = df.total_gas.astype(float)

# converting to datetime
df['date'] = pd.to_datetime(df['date'])
df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
df['week'] = df['date'].dt.day_name()
dates2 = np.array(df['date'])

df = df.drop(['date'], axis=1)
# df = df[['year', 'month', 'day', 'week', 'max_demand_gen', 'highest_gen', 'min_gen', 'day_peak_gen',
# 'eve_peak_gen', 'eve_peak_load_shedding', 'max_temp', 'total_gas', 'total_energy']] df.to_csv('rf_dataset.csv',
# index=False)

df = pd.get_dummies(df)
print(df.head(5))
df.to_csv('../csv_files/dataset_with_weekday.csv', index=False)
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
# The random state to 42 means the results will be the same each time I run the split for reproducible results
train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size=0.3, random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Instantiate model
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
rms = sqrt(mean_squared_error(test_labels, predictions))
print('rms', rms)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
print('MAPE', np.mean(mape))
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

plt.bar(range(len(feature_importances)), [val[1] for val in feature_importances], align='center')
plt.xticks(range(len(feature_importances)), [val[0] for val in feature_importances])
plt.xticks(rotation=90)
plt.ylabel('Relative Importance')
plt.xlabel('Features')
plt.savefig('../assets/var_imp_rf.png', bbox_inches='tight')
plt.show()

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# # Set the style
# plt.style.use('fivethirtyeight')
# # list of x locations for plotting
# x_values = list(range(len(importances)))
# # Make a bar chart
# plt.bar(x_values, importances, orientation='vertical')
# # Tick labels for x axis
# plt.xticks(x_values, feature_list, rotation='vertical')
# # Axis labels and title
# plt.ylabel('Relative Importance')
# plt.xlabel('Features')
# plt.title('Variable Importances')
# plt.savefig('var_imp.png', bbox_inches='tight')
# plt.show()

# Dates of training values
months = df[:, feature_list.index('month')]
days = df[:, feature_list.index('day')]
years = df[:, feature_list.index('year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# print(len(dates))
# print(dates)
# Dataframe with true values and dates

true_data = pd.DataFrame(data={'date': dates2, 'total_energy': labels})
# Dates of predictions
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]
# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]
# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})
# Plot the actual values
plt.plot(true_data['date'], true_data['total_energy'], 'k-', label='Actual', linewidth=1)
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', color='green', label='Prediction',
         markersize=3.0)
plt.xticks(rotation='30')
plt.legend()
# Graph labels
plt.xlabel('Time')
plt.ylabel('Electricity Consumption (MkWh)')
plt.savefig('../assets/random_forest', bbox_inches='tight')
plt.show()

# print(predictions_data.head())
# print(true_data.head())
# print(true_data.info())
# print(dates)
# print(test_dates)
# print(test_features)

plt.show()

# true_data = pd.DataFrame(data={'date': dates, 'total_energy': labels})
# df_prophet = true_data.rename(columns={'date': 'ds', 'total_energy': 'y'})
#
# m = Prophet()
# m.fit(df_prophet)
#
# future = m.make_future_dataframe(periods=365)
# future.tail()
#
# forecast = m.predict(future)
# fig1 = m.plot(forecast)
# fig2 = m.plot_components(forecast)

# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
#
export_graphviz(tree_small, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1,
                proportion=False, filled=True)
# Terminal commands to generate decision tree graph from dot file
# D:\RFLSTMHybridModel\src>set path=%path%;C:\Program Files (x86)\Graphviz2.38\bin
# D:\RFLSTMHybridModel\src>dot -Tpdf tree.dot -o ../assets/rf_decision_tree_graph.pdf

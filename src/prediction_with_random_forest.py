from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

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
x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, random_state=42)
print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Instantiate model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on training data
rf.fit(x_train, y_train)

# Use the forest's predict method on the test data
y_pred = rf.predict(x_test)
# Calculate the absolute errors
errors = abs(y_pred - y_test)
rms = sqrt(mean_squared_error(y_test, y_pred))
print('rms', rms)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
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
plt.savefig('../assets_lstm_2/var_imp_rf.png', bbox_inches='tight')
plt.show()

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=100, max_depth=3)
rf_small.fit(x_train, y_train)
# Extract the small tree
tree_small = rf_small.estimators_[5]
export_graphviz(tree_small, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1,
                proportion=False, filled=True)
# Terminal commands to generate decision tree graph from dot file
# D:\RFLSTMHybridModel\src>set path=%path%;C:\Program Files (x86)\Graphviz2.38\bin
# D:\RFLSTMHybridModel\src>dot -Tpdf tree.dot -o ../assets/rf_decision_tree_graph.pdf

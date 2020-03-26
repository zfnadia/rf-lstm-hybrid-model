import math
import time
import matplotlib.pyplot as plt2
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../csv_files/dataset_with_weekday.csv", index_col=0)


# df = df.drop(['eve_peak_load_shedding'], axis=1)
# df = df.drop(['max_temp'], axis=1)
# df = df.drop(['total_gas'], axis=1)

# prepare data for normalization
def normalize_data(df):
    min_max_scaler = MinMaxScaler()
    df['max_demand_gen'] = min_max_scaler.fit_transform(df.max_demand_gen.values.reshape(-1, 1))
    # df['max_demand_sub'] = min_max_scaler.fit_transform(df.max_demand_sub.values.reshape(-1, 1))
    df['highest_gen'] = min_max_scaler.fit_transform(df.highest_gen.values.reshape(-1, 1))
    df['min_gen'] = min_max_scaler.fit_transform(df.min_gen.values.reshape(-1, 1))
    df['day_peak_gen'] = min_max_scaler.fit_transform(df.day_peak_gen.values.reshape(-1, 1))
    df['eve_peak_gen'] = min_max_scaler.fit_transform(df.eve_peak_gen.values.reshape(-1, 1))
    df['eve_peak_load_shedding'] = min_max_scaler.fit_transform(df.eve_peak_load_shedding.values.reshape(-1, 1))
    df['max_temp'] = min_max_scaler.fit_transform(df.max_temp.values.reshape(-1, 1))
    df['total_gas'] = min_max_scaler.fit_transform(df.total_gas.values.reshape(-1, 1))
    df['total_energy'] = min_max_scaler.fit_transform(df.total_energy.values.reshape(-1, 1))
    return df


df = normalize_data(df)
# df['date'] = pd.to_datetime(df['date'])
print(df.head())


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)  # 5
    # Calling values returns a numpy.ndarray from a pandas series
    data = stock.values
    sequence_length = seq_len + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length])  # index : index + 22days

    result = np.array(result)
    row = round(0.9 * result.shape[0])  # 90% split
    train = result[:int(row), :]  # 90% date, all features

    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]

    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]


def build_model(layers):
    d = 0.2
    model = Sequential()

    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(1, kernel_initializer="uniform", activation='linear'))

    # adam = keras.optimizers.Adam(decay=0.2)
    start = time.time()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


window = 22
X_train, y_train, X_test, y_test = load_data(df, window)
print(X_train[0], y_train[0])

model = build_model([18, window, 1])
print(model.summary())
history = model.fit(X_train, y_train, batch_size=512, epochs=300, validation_split=0.1, verbose=1)

# print(X_test[-1])
diff = []
ratio = []
p = model.predict(X_test)
print(p.shape)
# for each data index in test data
for u in range(len(y_test)):
    # pr = prediction day u
    pr = p[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))


# Bug fixed at here, please update the denormalize function to this one
def denormalize(df, normalized_value):
    df = df['total_energy'].values.reshape(-1, 1)
    normalized_value = normalized_value.reshape(-1, 1)
    # return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new


from sklearn import metrics

print('MSE: ', metrics.mean_squared_error(y_test, p))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, p)))
print('Mape: ', metrics.mean_absolute_error(y_test, p, multioutput='uniform_average'))
newp = denormalize(df, p)
newy_test = denormalize(df, y_test)


def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.5f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.5f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


model_score(model, X_train, y_train, X_test, y_test)

# print(loss.mean_squared_error(y_train, y_test))
# print(loss.mean_absolute_error(y_train, y_test))
# print(loss.mean_absolute_percentage_error(y_train, y_test))
#
# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


plt2.plot(newp, color='green', label='Prediction', linewidth=1)
plt2.plot(newy_test, color='black', marker='.', label='Actual', linewidth=1)
plt2.legend(loc='best')
plt2.xlabel('Time [days]')
plt2.ylabel('Electricity Consumption (MkWh) \n[normalized values]')
plt2.savefig('../assets/lstm_predicted_vs_actual', bbox_inches='tight')
plt2.show()

plt2.figure(figsize=(8, 4))
plt2.plot(history.history['loss'], label='Train Loss')
plt2.plot(history.history['val_loss'], label='Test Loss')
plt2.title('model loss')
plt2.ylabel('Loss')
plt2.xlabel('Epochs')
plt2.legend(loc='upper right')
plt2.savefig('../assets/lstm_model_loss.png', bbox_inches='tight')
plt2.show()

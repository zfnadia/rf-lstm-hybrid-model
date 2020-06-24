import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras.backend as K
from pandas.plotting import register_matplotlib_converters
from pylab import rcParams
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from sklearn import metrics

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
""" 
parse_dates is used by Pandas to automatically recognize dates.
Pandas implicitly recognizes the format by agr infer_datetime_format=True
https://stackoverflow.com/questions/17465045/can-pandas-automatically-recognize-dates
"""

df = pd.read_csv('../csv_files/main_dataset.csv', parse_dates=['date'], infer_datetime_format=True)

# df['day_of_week'], df['day_of_month'], df['month'], df['is_weekend'],
df['day_of_week'] = df['date'].dt.dayofweek  # monday = 0, sunday = 6
df['day_of_month'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['is_weekend'] = 0  # Initialize the column with default value of 0
df.loc[df['day_of_week'].isin([4, 5]), 'is_weekend'] = 1  # 4 and 5 correspond to Fri and Sat


def season_of_date(date):
    year = str(date.year)
    seasons = {'summer': pd.date_range(start='01/03/' + year, end='29/06/' + year),
               'rainy': pd.date_range(start='30/06/' + year, end='29/09/' + year),
               'winter': pd.date_range(start='30/09/' + year, end='31/12/' + year)}
    if date in seasons['summer']:
        return 0
    if date in seasons['rainy']:
        return 1
    if date in seasons['winter']:
        return 2
    else:
        return 2


## generate df['season']
# "season" - *category field meteorological seasons: 0-summer ; 1-rainy; 2-winter*
df['season'] = df['date'].map(season_of_date)

# convert the date format in the timestamp column from yyyy-dd-mm to yyyy-mm-dd
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

df.to_csv('../csv_files/main_dataset_2.csv', index=False)
df = pd.read_csv('../csv_files/main_dataset_2.csv', parse_dates=['date'], index_col="date",
                 infer_datetime_format=True)
print(df.head())

## timestamp_vs_energy plot
# index_vs_energy_plot = sns.lineplot(x=df.index, y="total_energy", data=df, legend='brief')
# index_vs_energy_plot.set(xlabel='timestamp', ylabel='electricity_consumption (MKWh)')
# plt.savefig('../assets_lstm_2/day_vs_total_energy.png', bbox_inches='tight')
# plt.show()

# # resample by each month
# # take sum for each month to find seasonality, consumption rises during rainy and winter season
# df_by_month = df.resample('M').sum()
# index_vs_energy_plot = sns.lineplot(x=df_by_month.index, y="total_energy", data=df_by_month)
# index_vs_energy_plot.set(xlabel='timestamp', ylabel='electricity_consumption (MKWh)')
# plt.savefig('../assets_lstm_2/month_vs_total_energy.png', bbox_inches='tight')
# plt.show()

## season plot
# x_dates = df.index.strftime('%Y-%m-%d').sort_values().unique()
# season_plot = sns.pointplot(data=df, x=x_dates, y='total_energy', hue='season')
# season_plot.set_xticklabels(x_dates, rotation=30)
# season_plot.set(xlabel='timestamp', ylabel='electricity_consumption (MKWh)')
# leg_handles = season_plot.get_legend_handles_labels()[0]
# season_plot.legend(leg_handles, ['summer', 'rainy', 'winter'], title='seasons')
# # show ticks every 2 months along x axis
# xticks = season_plot.xaxis.get_major_ticks()
# month = None
# take_next = False
#
# for tick in xticks:
#   m = tick.label.get_text()[5:7]
#   if month != m and take_next:
#     month = m
#     take_next = False
#   elif month != m:
#     month = m
#     take_next = True
#     tick.set_visible(False)
#   else:
#     tick.set_visible(False)
#
# plt.savefig('../assets_lstm_2/seasonality_by_season.png', bbox_inches='tight')
# plt.show()

## is_weekend plot
# x_dates = df.index.strftime('%Y-%m-%d').sort_values().unique()
# season_plot = sns.pointplot(data=df, x=x_dates, y='total_energy', hue='is_weekend')
# season_plot.set_xticklabels(x_dates, rotation=30)
# season_plot.set(xlabel='timestamp', ylabel='electricity_consumption (MKWh)')
# leg_handles = season_plot.get_legend_handles_labels()[0]
# season_plot.legend(leg_handles, ['weekday', 'weekend'], title='is_weekday')
# # show ticks every 2 months along x axis
# xticks = season_plot.xaxis.get_major_ticks()
# month = None
# take_next = False
#
# for tick in xticks:
#   m = tick.label.get_text()[5:7]
#   if month != m and take_next:
#     month = m
#     take_next = False
#   elif month != m:
#     month = m
#     take_next = True
#     tick.set_visible(False)
#   else:
#     tick.set_visible(False)
#
# plt.savefig('../assets_lstm_2/seasonality_by_weekend.png', bbox_inches='tight')
# plt.show()

# 90 percent of the data is used for training
train_size = int(len(df) * 0.9)
# rest of the data is used for test
test_size = len(df) - train_size
# split actual dataset into test and train variables
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))
# (1300, 14) (145, 14), 1300 examples for train and 145 examples for test
print(train.shape, test.shape)

# scaling the features for better result
f_columns = ['max_demand_gen', 'highest_gen', 'min_gen', 'day_peak_gen', 'eve_peak_gen']

# f_columns = ['max_demand_gen', 'highest_gen', 'min_gen', 'day_peak_gen', 'eve_peak_gen', 'eve_peak_load_shedding',
#              'max_temp', 'total_gas', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'season']

f_transformer = RobustScaler()
total_energy_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
total_energy_transformer = total_energy_transformer.fit(train[['total_energy']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['total_energy'] = total_energy_transformer.transform(train[['total_energy']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['total_energy'] = total_energy_transformer.transform(test[['total_energy']])


# get sequences of data


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# take last 30 days to predict the data of the next day
time_steps = 10

# reshape to [samples, time_steps, n_features]
# (1270, 30, 14) (1270,)
X_train, y_train = create_dataset(train, train.total_energy, time_steps)
X_test, y_test = create_dataset(test, test.total_energy, time_steps)

# reshape to [samples, time_steps, n_features]
# (1270, 30, 14) (1270,)
print(X_train.shape, y_train.shape)


def percentage_difference(y_true, y_pred):
    return K.mean(abs(y_pred/y_true - 1) * 100)

model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128,  # units is number of neurons
            input_shape=(X_train.shape[1], X_train.shape[2])  # 30, 14
        )
    )
)
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', percentage_difference])

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=512,
    validation_split=0.1,
    shuffle=False
)

plt.clf()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='training loss', linewidth=3)
plt.plot(epochs, val_loss, 'y', label='validation loss', linewidth=3)
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('../assets_lstm_2/lstm_loss_imp_feat.png', bbox_inches='tight')
plt.show()

y_pred = model.predict(X_test)
y_train_inv = total_energy_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = total_energy_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = total_energy_transformer.inverse_transform(y_pred)

print('MSE: ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('MAPE: ', metrics.mean_absolute_error(y_test, y_pred, multioutput='uniform_average'))

plt.clf()
plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), linewidth=3, marker='.',
         markersize='12', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', linewidth=3, marker='.',
         markersize='12', label="prediction")
plt.ylabel('electricity consumption (MKWh)')
plt.xlabel('time step')
plt.legend()
# plt.savefig('../assets_lstm_2/lstm_test_vs_train_all_feat.png', bbox_inches='tight')
plt.savefig('../assets_lstm_2/lstm_test_vs_train_imp_feat.png', bbox_inches='tight')
plt.show()

plt.clf()
plt.plot(y_test_inv.flatten(), marker='.', label="true", linewidth=3, markersize='12')
plt.plot(y_pred_inv.flatten(), 'r', marker='.', label="prediction", linewidth=3, markersize='12')
plt.ylabel('electricity Consumption (MKWh)')
plt.xlabel('time step')
plt.legend()
# plt.savefig('../assets_lstm_2/total_test_vs_train_all_feat.png', bbox_inches='tight')
plt.savefig('../assets_lstm_2/total_test_vs_train_imp_feat.png', bbox_inches='tight')
plt.show()

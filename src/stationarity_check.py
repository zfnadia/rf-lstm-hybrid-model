import pandas as pd
import numpy as np
from numpy import log
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('../csv_files/main_dataset_lstm_2.csv', header=0, index_col=0, squeeze=True)
print(df.describe())

df = df.drop('max_demand_gen', axis=1)
df = df.drop('highest_gen', axis=1)
df = df.drop('min_gen', axis=1)
df = df.drop('day_peak_gen', axis=1)
df = df.drop('eve_peak_gen', axis=1)
df = df.drop('max_temp', axis=1)
df = df.drop('eve_peak_load_shedding', axis=1)
df = df.drop('total_gas', axis=1)
print(df.head())

df.plot()
pyplot.savefig('../assets_lstm_2/dataset.png', bbox_inches='tight')
pyplot.show()
df.hist()
pyplot.savefig('../assets_lstm_2/histogram.png', bbox_inches='tight')
pyplot.show()

X = df.values
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))


X = df['total_energy'].values
X = log(X)
print(X)
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

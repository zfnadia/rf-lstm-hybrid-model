
import pandas as pd
import datetime

# start_date = date(year=2015, month=5, day=31)
start_date = '2015-05-31'
start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_date = '2018-07-04'
end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
step = datetime.timedelta(days=1)

df = pd.read_csv('thesis_data_pre_thesis_2.csv')

df.drop(df.iloc[:, 13:23], inplace=True, axis=1)
df["date"] = df.apply(lambda x: start_date + step, axis=1)
df.to_csv('rf_dataset.csv', index=False)
print(df.head())


# prepare data for normalization

# def normalize_data(df):
#     min_max_scaler = MinMaxScaler()
#     df['max_demand_gen'] = min_max_scaler.fit_transform(df.max_demand_gen.values.reshape(-1, 1))
#     df['max_demand_sub'] = min_max_scaler.fit_transform(df.max_demand_sub.values.reshape(-1, 1))
#     df['highest_gen'] = min_max_scaler.fit_transform(df.highest_gen.values.reshape(-1, 1))
#     df['min_gen'] = min_max_scaler.fit_transform(df.min_gen.values.reshape(-1, 1))
#     df['day_peak_gen'] = min_max_scaler.fit_transform(df.day_peak_gen.values.reshape(-1, 1))
#     df['eve_peak_gen'] = min_max_scaler.fit_transform(df.eve_peak_gen.values.reshape(-1, 1))
#     df['eve_peak_load_shedding'] = min_max_scaler.fit_transform(df.eve_peak_load_shedding.values.reshape(-1, 1))
#     df['water_level_kaptai'] = min_max_scaler.fit_transform(df.water_level_kaptai.values.reshape(-1, 1))
#     df['rule_curve'] = min_max_scaler.fit_transform(df.rule_curve.values.reshape(-1, 1))
#     df['max_temp'] = min_max_scaler.fit_transform(df.max_temp.values.reshape(-1, 1))
#     df['total_gas'] = min_max_scaler.fit_transform(df.total_gas.values.reshape(-1, 1))
#     df['total_energy'] = min_max_scaler.fit_transform(df.total_energy.values.reshape(-1, 1))
#     return df
#
#
# df = normalize_data(df)
# # df = df.drop(['date'], axis=1)
# print(df.head())
print(df.info())
# df_train_features = df.drop(['total_energy'], axis=1)
# X = df_train_features
# y = df['total_energy']
# print(X.head())
# print(y.head())
#
# # create an instance for tree feature selection
# tree_clf = ExtraTreesClassifier()
#
# # fit the model
# tree_clf.fit(X, y)
# #
# # # Preparing variables
# importances = tree_clf.feature_importances_
# feature_names = df_train_features.columns.tolist()
# #
# feature_imp_dict = dict(zip(feature_names, importances))
# sorted_features = sorted(feature_imp_dict.items(), key=np.operator.itemgetter(1), reverse=True)
# #
# indices = np.argsort(importances)[::-1]
# #
# # Print the feature ranking
# print("Feature ranking:")
#
# for f in range(X.shape[1]):
#     print("feature %d : %s (%f)" % (indices[f], sorted_features[f][0], sorted_features[f][1]))
#
# # Plot the feature importances of the forest
# plt.figure(0)
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#         color="r", align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()

# # def normalize_data(df):
# #     min_max_scaler = preprocessing.MinMaxScaler()
# #     df['opening_price'] = min_max_scaler.fit_transform(df.opening_price.values.reshape(-1, 1))
# #     df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
# #     df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
# #     df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1, 1))
# #     df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1, 1))
# #     return df
#

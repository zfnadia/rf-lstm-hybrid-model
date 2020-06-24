import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# open the csv, chose company_N, where N = {A, B, C or D}
df = pd.read_csv('../csv_files/main_dataset_lstm_2.csv')
print(df.head())
# for box plots
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['total_energy'] = pd.to_numeric(df['total_energy'], errors='coerce')
df = df.dropna(subset=['total_energy'])
df['year'] = df['timestamp'].apply(lambda x: x.year)
df['quarter'] = df['timestamp'].apply(lambda x: x.quarter)
df['month'] = df['timestamp'].apply(lambda x: x.month)
df['day'] = df['timestamp'].apply(lambda x: x.day)
df = df.loc[:, ['timestamp', 'total_energy', 'year', 'quarter', 'month', 'day']]
df.sort_values('timestamp', inplace=True, ascending=True)
df = df.reset_index(drop=True)
df['weekday'] = pd.to_datetime(df['timestamp']).dt.dayofweek  # monday = 0, sunday = 6
df['weekend_indi'] = 0  # Initialize the column with default value of 0
df.loc[df['weekday'].isin([4, 5]), 'weekend_indi'] = 1  # 5 and 6 correspond to Sat and Sun
print(df.shape)
print(df.timestamp.min())
print(df.timestamp.max())
print(df.tail(5))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.subplots_adjust(wspace=0.2)
sns.boxplot(x="year", y="total_energy", data=df)
plt.xlabel('years')
plt.ylabel('electricity_onsumption (MKWh)')
sns.despine(left=True)
plt.tight_layout()

plt.subplot(1, 2, 2)
sns.boxplot(x="quarter", y="total_energy", data=df)
plt.xlabel('quarters')
plt.ylabel('electricity_onsumption (MKWh)')
sns.despine(left=True)
plt.tight_layout()
plt.savefig('../assets_lstm_2/yearly_quarterly_new.png', bbox_inches='tight')
plt.show()

# dic={0:'Weekend',1:'Weekday'}
# df['Day'] = df.weekend_indi.map(dic)
#
# a=plt.figure(figsize=(9,4))
# plt1=sns.boxplot('year','total_energy',hue='Day',width=0.6,fliersize=3,
#                     data=df)
# a.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
# sns.despine(left=True, bottom=True)
# plt.xlabel('Years')
# plt.ylabel('Consumption (MkWh)')
# plt.tight_layout()
# plt.legend().set_visible(False)
# plt.savefig('weekday_vs_weekend.png', bbox_inches='tight')
# plt.show()


# df = df.drop(['max_demand_sub', 'rule_curve'], axis=1)
# print(df.head())
# # # set date as index
# df2 = pd.DataFrame(data = {'weekday': df['day_of_week'], 'energy': df['total_energy']})
# df2 = df2.groupby('weekday')
#
# ax = df2.plot()
#
# plt.tight_layout()
# plt.show()

# df.set_index('date', inplace=True)
# # keep only the 'Close' column
# df = df['max_temp']
# print(df.head())
#
# plt.figure(figsize=(15, 10))
# plt.plot(df, label='Max Temperature (°C)', color='green')
# plt.legend(loc='best')
#
# plt.xlabel('Date')
# plt.ylabel('Max Temperature (°C)')
# plt.savefig('date_vs_max_temp.png', bbox_inches='tight')
# plt.show()

# fig = plt.figure(figsize=(15, 10))
# ax1 = fig.add_subplot(231)
# ax1.set_title('Energy consumption during Ramadan 2015 (June 17 - July 16)')
# ax1.plot(df['date'], df['total_energy'], color='green')
# ax1.set_xlabel('date')
# ax1.set_ylabel('total energy (MKWh)')
# plt.xticks(rotation = '60')
# plt.savefig('2015_ramadan.png', bbox_inches='tight')
# plt.show()

# plt.figure(figsize=(20, 8))
# plt.plot(df['date'], df['max_temp'], 'b-', label = 'max temp (°C)')
# plt.plot(df['date'], df['total_energy'], 'r-', label = 'total energy (MKWh)')
# plt.xlabel('date'); plt.ylabel(''); plt.title('Energy consumption rates and max temperatures during Ramadan 2018 (May 16 - June 14)')
# plt.legend()
# plt.xticks(rotation = '60')
# plt.savefig('2018_ramadan_power.png', bbox_inches='tight')
# plt.show()
# # energy = df['total_energy']
# # max_temp = df['max_temp']
# # plt.scatter(max_temp, energy, edgecolors='r')
# # plt.xlabel('Max_Temperature (°C)')
# # plt.ylabel('Total Energy (MKWh)')
# # plt.savefig('corel_load_max_temp.png', bbox_inches='tight')
# # plt.show()
#


# names = ['Historical Load']
# datum = [df]
# fig = plt.figure(figsize=(20, 25))
# for i, data in enumerate(datum):
#     ax = fig.add_subplot(int(str(32) + str(i + 1)))
#     null_pct = data.isnull().sum()
#     sns.barplot(null_pct.values, null_pct.index, ax=ax)
#
#     for idx, value in enumerate(null_pct.values):
#         ax.text(-0.055, idx, "{:.2f}%".format(float(value) / len(data) * 100), fontsize=18)
#     ax.set_xlabel("number of missing data", fontsize=18)
#     ax.set_ylabel("Column Name", fontsize=18)
#     ax.set_title("Missing data for {} dataframe".format(names[i]), fontsize=20)
#
# plt.show()
# plt.savefig('missing_data.png', bbox_inches='tight')
#
# temp = df[['total_energy', 'date']]
# temp.loc[:, 'date'] = pd.to_datetime(temp['date'])
# temp.loc[:,'Month'] = temp['date'].dt.month
# count_mean = temp.groupby('Month')['total_energy'].agg(['count','mean']).sort_index()
# count_mean.rename(columns={'count':'Counts','mean':'Average'},inplace=True)
# fig = plt.figure(figsize=(10,12))
# for idx,col in enumerate(count_mean.columns):
#     ax = fig.add_subplot(int(str(21)+str(idx+1)))
#     sns.barplot(x=count_mean.index,y=col,data=count_mean,ax=ax)
#     ax.set_title("Distribution of Power Consumption {} Among Months".format(col),fontsize=16)
#     plt.ylabel("Power Consumed (MKWh)")
# plt.savefig('monthly_plot.pdf', bbox_inches='tight')
# plt.show()
#
#
# temp = df[['total_energy', 'date']]
# temp.loc[:, 'date'] = pd.to_datetime(temp['date'])
# temp.loc[:,'Month'] = temp['date'].dt.weekday
# count_mean = temp.groupby('Month')['total_energy'].agg(['mean']).sort_index()
# count_mean.rename(columns={'mean':'Average'})
# count_mean.rename(index={0:'Friday', 1:'Saturday', 2:'Sunday', 3:'Monday', 4:'Thuesday', 5:'Wednesday', 6:'Thursday'}, inplace=True)
# fig = plt.figure(figsize=(10,12))
# for idx,col in enumerate(count_mean.columns):
#     ax = fig.add_subplot(int(str(21)+str(idx+1)))
#     sns.barplot(count_mean.index,count_mean[col],ax=ax)
#     ax.set_title("Distribution of {} Power Consumption Among Weekdays".format(col),fontsize=16)
#     plt.ylabel("Power Consumed (MKWh)")
#     plt.xlabel("Day of Week")
#
# plt.savefig('weekday_plot.pdf', bbox_inches='tight')
# plt.show()

# temp and gas scatter plot
# plt.figure(figsize=(14, 5))
# plt.subplot(1, 2, 1)
# plt.subplots_adjust(wspace=0.2)
# energy = df['total_energy']
# max_temp = df['max_temp']
# total_gas = df['total_gas']
# plt.scatter(max_temp, energy, edgecolors='r')
# plt.xlabel('temperature (°C)')
# plt.ylabel('electricity_consumption (MKWh)')
# sns.despine(left=True)
# plt.tight_layout()
#
# plt.subplot(1, 2, 2)
# plt.scatter(total_gas, energy, edgecolors='r')
# plt.xlabel(' gas supplied (MMCFD)')
# plt.ylabel('electricity_consumption (MKWh)')
# sns.despine(left=True)
# plt.tight_layout()
# plt.savefig('../assets_lstm_2/scatter_temp_gas_new.png', bbox_inches='tight')
# plt.show()

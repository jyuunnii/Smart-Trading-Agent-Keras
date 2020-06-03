"""
# dependencies
pandas
matplotlib.pyplot
seaborn
sklearn
numpy
"""

# In[1]:
import pandas as pd

trade = pd.read_csv('data/2018-05-trade.csv')

# In[2]:
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

# In[3]:
side_buy = trade[trade['side']==0]
side_buy

tcost = 0
for cost in side_buy:
    tcost = side_buy.amount.sum()

tcost ### -1283164019


side_sell = trade[trade['side']==1]
side_sell

tincome = 0
for income in side_sell:
    tincome = side_sell.amount.sum()

tincome ### 1273657117

tprofit = tincome - tcost ### -9506902

# In[4]:
## periods = trade['timestamp'].map(lambda x:x.split(':')[0].strip()) #### 2018-05-01 01
trade['timestamp'] = trade['timestamp'].map(lambda x:x.split('8-')[1].split(':')[0].strip()) #### 05-01 01

periods = trade['timestamp'].unique()
periods.size ### 45


temp = trade

td1 = temp[['timestamp', 'side', 'tx']] # extract some columns from dataframe
td2 = td1.groupby(['timestamp','side']).size().reset_index(name='counts')
td3 = td2.reset_index().groupby(['timestamp', 'side'])['counts'].aggregate('first').unstack()
td3['total'] = td3.sum(axis=1) # add a column 
td3 = td3.rename(columns={0:'buy', 1:'sell'})

td3.plot()
plt.figure(figsize=(16,8))
plt.title('Time-series Transaction Graph')
plt.xlabel('time')
plt.ylabel('transactions')
plt.show()

# In[5]:
orderbook1 = pd.read_csv('data/2018-05-01-orderbook.csv')
orderbook2 = pd.read_csv('data/2018-05-02-orderbook.csv')

orderbook1['timestamp'] = orderbook1['timestamp'].map(lambda x:x.split('8-')[1].split('.')[0].strip()) #### 05-01 00:00:00
orderbook2['timestamp'] = orderbook2['timestamp'].map(lambda x:x.split('8-')[1].split('.')[0].strip()) #### 05-02 00:00:00

nb1 = orderbook1.groupby('timestamp').agg(top_bid_price=('price', lambda x:list(x)[14]), top_ask_price = ('price', lambda x: list(x)[15]))
nb1['MidPrice'] = nb1.mean(axis=1) # add a column 

nb2 = orderbook1.groupby(['timestamp', 'type']).mean()
nb2.reset_index().groupby(['timestamp', 'type'])['quantity','price'].aggregate('first').unstack()
nb2.rename(columns={0:'bid', 1:'ask'})







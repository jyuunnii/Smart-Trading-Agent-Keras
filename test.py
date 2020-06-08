"""
# dependencies
pandas
matplotlib.pyplot
seaborn
sklearn
numpy

conda info --env (project list)
conda activate [PROJECT NAME] (activation)
ipython
"""

# In[1]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# In[2]:
trade = pd.read_csv('data/2018-05-trade.csv')

side_buy = trade[trade['side']==0]
side_buy ### total 398 data

tcost = 0
for cost in side_buy:
    tcost = side_buy.amount.sum()

tcost ### -1283164019


side_sell = trade[trade['side']==1]
side_sell ### total 514 data

tincome = 0
for income in side_sell:
    tincome = side_sell.amount.sum()

tincome ### 1273657117

tprofit = tincome + tcost ### -9506902

# In[3]:
## periods = trade['timestamp'].map(lambda x:x.split(':')[0].strip()) #### 2018-05-01 01
trade['timestamp'] = trade['timestamp'].map(lambda x:x.split('8-')[1].split(':')[0].strip()) #### 05-01 01

periods = trade['timestamp'].unique()
periods.size ### 45


temp = trade
temp['tx'] = 1 # new column for calculation
td1 = temp[['timestamp', 'side', 'tx']] # extract needed columns from original dataframe
td2 = td1.groupby(['timestamp','side']).size().reset_index(name='counts')
td3 = td2.reset_index().groupby(['timestamp', 'side'])['counts'].aggregate('first').unstack().rename(columns={0:'Buy', 1:'Sell'})
td3['Total'] = td3.sum(axis=1) # add a column 

td3.plot()
plt.figure(figsize = (16,8))
plt.title('Time-series Transaction Graph')
plt.xlabel('time')
plt.ylabel('transactions')
plt.show()

# In[4]:
orderbook1 = pd.read_csv('data/2018-05-01-orderbook.csv')

orderbook1['timestamp'] = orderbook1['timestamp'].map(lambda x:x.split('8-')[1].split('.')[0].strip()) #### 05-01 00:00:00

nb1 = orderbook1.groupby('timestamp').agg(top_bid_price=('price', lambda x:list(x)[14]), top_ask_price = ('price', lambda x: list(x)[15]))
nb1['mid_price'] = nb1.mean(axis=1) # add a column 
nb1.reset_index(level=['timestamp'], inplace = True)

nb2 = orderbook1.groupby(['timestamp', 'type']).mean()
temp1 = nb2.reset_index().groupby(['timestamp', 'type'])['quantity'].aggregate('first').unstack().rename(columns={0:'bidQty', 1:'askQty'})
temp1.astype(int)
temp1.reset_index(level=['timestamp'], inplace = True)
temp2 = nb2.reset_index().groupby(['timestamp', 'type'])['price'].aggregate('first').unstack().rename(columns={0:'bidPx', 1:'askPx'})
temp2.astype(int)
temp2.reset_index(level=['timestamp'], inplace = True)


nb3 = pd.merge(temp1, temp2)
nb3['mid_price'] = nb1['mid_price']
nb3

# In[5]:
import numpy as np

def bp(aQ,bQ,aP,bP): 
    return (((aQ*bP)/bQ)+((bQ*aP)/aQ))/(bQ+aQ)

nb3['book_price'] = np.vectorize(bp)(nb3['askQty'], nb3['bidQty'], nb3['askPx'], nb3['bidPx']).astype(int)
nb3['book_feature'] = np.subtract(nb3['book_price'], nb3['mid_price'])


newbook1 = nb3[['timestamp', 'book_price', 'mid_price', 'book_feature']]

#newbook1_DF = pd.DataFrame(newbook1)
#newbook1_DF.to_csv('data/2018-05-01-newbook.csv', header = True, index= False)


# In[6]: Repeat for orderbook2
orderbook2 = pd.read_csv('data/2018-05-02-orderbook.csv')

orderbook2['timestamp'] = orderbook2['timestamp'].map(lambda x:x.split('8-')[1].split('.')[0].strip()) #### 05-02 00:00:00

nb1 = orderbook2.groupby('timestamp').agg(top_bid_price=('price', lambda x:list(x)[14]), top_ask_price = ('price', lambda x: list(x)[15]))
nb1['mid_price'] = nb1.mean(axis=1) # add a column 
nb1.reset_index(level=['timestamp'], inplace = True)

nb2 = orderbook2.groupby(['timestamp', 'type']).mean()
temp1 = nb2.reset_index().groupby(['timestamp', 'type'])['quantity'].aggregate('first').unstack().rename(columns={0:'bidQty', 1:'askQty'})
temp1.astype(int)
temp1.reset_index(level=['timestamp'], inplace = True)
temp2 = nb2.reset_index().groupby(['timestamp', 'type'])['price'].aggregate('first').unstack().rename(columns={0:'bidPx', 1:'askPx'})
temp2.astype(int)
temp2.reset_index(level=['timestamp'], inplace = True)


nb3 = pd.merge(temp1, temp2)
nb3['mid_price'] = nb1['mid_price']
nb3['book_price'] = np.vectorize(bp)(nb3['askQty'], nb3['bidQty'], nb3['askPx'], nb3['bidPx']).astype(int)
nb3['book_feature'] = np.subtract(nb3['book_price'], nb3['mid_price'])

newbook2 = nb3[['timestamp', 'book_price', 'mid_price', 'book_feature']]
#newbook2_DF = pd.DataFrame(newbook2)
#newbook2_DF.to_csv('data/2018-05-02-newbook.csv', header = True, index= False)


#In[7]:
tnewbook = pd.concat([newbook1, newbook2])
tnewbook_DF = pd.DataFrame(tnewbook)
tnewbook_DF.to_csv('data/2018-05-trade.csv', header = True, index = False)
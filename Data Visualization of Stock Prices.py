# # Part 1
# 
# 
# ## Get the Data
# ### The Imports
# 
# In[1]:
#!pip install pandas-datareader
#conda install -c anaconda pandas-datareader 
#!pip install --upgrade pandas
import pandas as pd
import numpy as np
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data, wb

import datetime
from datetime import date
get_ipython().run_line_magic('matplotlib', 'inline')

# ## Data
# 
# We need to get data using pandas datareader. We will get stock information for the following companies:
# * Amazon
# * Facebook
# * Google
# * Microsoft
# * Twitter
# * Apple
# 
# 
# ** Figure out how to get the stock data from Jan. 1st 2020 until now for each of these companies. Set each company to be a separate dataframe, with the variable name for that bank being its ticker symbol. This will involve a few steps:**
# 1. Use datetime to set start and end datetime objects.
# 2. Figure out the ticker symbol for each company.
# 2. Figure out how to use datareader to grab info on the stock.
# 
# In[2]:


start = datetime.datetime(2020, 1, 1)

end = datetime.datetime(2022, 10, 1)


# In[3]:


# Amazon
Amazon = data.DataReader("AMZN", 'yahoo', start, end)

# Facebook
Facebook = data.DataReader("Meta", 'yahoo', start, end)

# Google
Google = data.DataReader("GOOG", 'yahoo', start, end)

# Microsoft
Microsoft = data.DataReader("MSFT", 'yahoo', start, end)

# Twitter
Twitter = data.DataReader("TWTR", 'yahoo', start, end)

# Apple
Apple = data.DataReader("AAPL", 'yahoo', start, end)


# In[4]:


Amazon.reset_index(inplace=True)
Amazon["Date"].value_counts()

Facebook.reset_index(inplace=True)
Facebook["Date"].value_counts()

Google.reset_index(inplace=True)
Google["Date"].value_counts()

Microsoft.reset_index(inplace=True)
Microsoft["Date"].value_counts()

Twitter.reset_index(inplace=True)
Twitter["Date"].value_counts()

Apple.reset_index(inplace=True)
Apple["Date"].value_counts()

Apple.head()

# In[6]:

Amazon["Company"]='Amazon'
Facebook["Company"]='Facebook'
Microsoft["Company"]='Microsoft'
Twitter["Company"]='Twitter'
Apple["Company"]='Apple'
Google["Company"]='Google'


# ##### Draw the closing prices of Amazon

# In[7]:


import matplotlib.pyplot as plt

# your code here

plt.plot(Amazon['Date'],Amazon['Close'])
plt.title('Amazon\' Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')

plt.tight_layout()
plt.savefig("normalvars.png", dpi=150)


# ##### Append all the data sets - these six tables

# In[8]:

df = pd.concat([Amazon, Facebook, Microsoft, Twitter, Apple, Google], ignore_index=True)
df

# ** Derive the average closing price for each company, and then plot the average closing price using a line plot for each company using matplotlib or other visualization libraries (e.g. plotly and seaborn). **

# In[9]:

company_set = set()

for i in df.Company:
    company_set.add(i)


# In[10]:


# Find the average closing price for each company
def find_avg_closing_price(company):
    return df[df['Company'] == company]['Close'].rolling(1).mean()


# In[11]:


# plot the average closing price for each company 
def plot_image(company, index):
    sub = fig.add_subplot(3,2,index)
    sub.set_xlabel('date')
    sub.set_ylabel('price')
    sub.set_title(company)
    #sub.plot(find_avg_closing_price(company))
    sub.plot(df[df['Company'] == company]['Date'],find_avg_closing_price(company))


# In[12]:

fig = plt.figure(figsize=(10,6))

index = 1
for i in company_set:
    plot_image(i, index)
    index = index + 1
    
plt.tight_layout()
plt.savefig("normalvars.png", dpi=150)


# In[13]:

# plot the average closing price for each company, all in one
for i in company_set:
    plt.plot(df[df['Company'] == i]['Date'],find_avg_closing_price(i), label = i)
    plt.legend()


plt.tight_layout()
plt.savefig("normalvars.png", dpi=150)


# ** Generate a spread chart using the adj closing price for the company Amazon and Facebook. **

# In[54]:

# your code here
Amazon_adj_price = df[df['Company'] == 'Amazon']['Adj Close'].to_list()
Facebook_adj_price = df[df['Company'] == 'Facebook']['Adj Close'].to_list()
date_list = Facebook['Date'].to_list()

df22 = pd.DataFrame({'Date': date_list, 'Amazon': Amazon_adj_price, 'Facebook': Facebook_adj_price})
df22 = df22.set_index('Date')
df22[['Amazon','Facebook']].iplot(kind='spread')


# ** Create a new dataframe called returns. This dataframe will contain the returns for each company's stock. returns are typically defined by:**
# 
# $$r_t = \frac{p_t - p_{t-1}}{p_{t-1}} = \frac{p_t}{p_{t-1}} - 1$$

# ** We can use pandas pct_change() method on the Close column to create a new dataframe representing this return value. Use .groupby().**

# In[15]:


# your code here
df2 =  df.groupby(['Company']).Close.pct_change()
returns = df2.to_frame()
returns['Company'] = df['Company']
returns['Date'] = df['Date']
returns['Return'] = df['Close'].pct_change()
returns.head()


# ** Using this returns DataFrame, plot the distribution of single day returns of each company across the entire time period ?**

# In[17]:

def plot_distribution(company,index):
    sub = fig.add_subplot(3,2,index)
    #new_df[new_df['Company'] == company]['Close'].hist(bins=50)
    sub.plot(returns[returns['Company'] == company]['Date'],returns[returns['Company'] == company]['Return'])
    sub.set_xlabel('Return')
    sub.set_ylabel('Probability')
    sub.set_title(company)


# In[18]:


fig = plt.figure(figsize=(15,7))

index = 1
for i in company_set:
    plot_distribution(i, index)
    index = index + 1
    
plt.tight_layout()
plt.savefig("normalvars.png", dpi=150)


# ** Choose a figure to visualize the standard deviation of the returns over the entire time period **


# In[20]:


def stdCal(lst):
    mean = sum(lst) / len(lst)
    variance = sum([((x - mean) ** 2) for x in lst]) / len(lst)
    res = variance ** 0.5
    return round(res, 3)

Amazon_close_returns = returns[returns['Company'] == 'Amazon']['Close'].tolist()[1:]
std_Amazon = stdCal(Amazon_close_returns) 

Facebook_close_returns = returns[returns['Company'] == 'Facebook']['Close'].tolist()[1:]
std_Facebook = stdCal(Facebook_close_returns) 

Microsoft_close_returns = returns[returns['Company'] == 'Microsoft']['Close'].tolist()[1:]
std_Microsoft = stdCal(Microsoft_close_returns) 

Twitter_close_returns = returns[returns['Company'] == 'Twitter']['Close'].tolist()[1:]
std_Twitter = stdCal(Twitter_close_returns) 

Apple_close_returns = returns[returns['Company'] == 'Apple']['Close'].tolist()[1:]
std_Apple = stdCal(Apple_close_returns) 

Google_close_returns = returns[returns['Company'] == 'Google']['Close'].tolist()[1:]
std_Google = stdCal(Google_close_returns) 

dct2 = {'Amazon': std_Amazon, 'Facebook': std_Facebook, 'Microsoft': std_Microsoft, 'Twitter': std_Twitter, 'Apple': std_Apple, 'Google': std_Google}
fig, ax = plt.subplots()
plt.plot(pd.Series(dct2.values()))
ax.set_xticklabels(['','Amazon', 'Facebook', 'Microsoft', 'Twitter', 'Apple', 'Google'])
ax.set_title('Standard Deviation Line Plot')


# ** Create a density plot using any library you like to visualize the return for each company in 2020**

# In[22]:


Amazon_2020 = returns[(returns['Company'] == 'Amazon') & (returns.Date.dt.year == 2020)]['Close'].tolist()[1:]
Facebook_2020 = returns[(returns['Company'] == 'Facebook') & (returns.Date.dt.year == 2020)]['Close'].tolist()[1:]
Microsoft_2020 = returns[(returns['Company'] == 'Microsoft') & (returns.Date.dt.year == 2020)]['Close'].tolist()[1:]
Twitter_2020 = returns[(returns['Company'] == 'Twitter') & (returns.Date.dt.year == 2020)]['Close'].tolist()[1:]
Apple_2020 = returns[(returns['Company'] == 'Apple') & (returns.Date.dt.year == 2020)]['Close'].tolist()[1:]
Google_2020 = returns[(returns['Company'] == 'Google') & (returns.Date.dt.year == 2020)]['Close'].tolist()[1:]

df_2020 = pd.DataFrame({'Amazon': Amazon_2020, 'Facebook': Facebook_2020, 'Microsoft': Microsoft_2020, 'Twitter': Twitter_2020, 'Apple': Apple_2020, 'Google': Google_2020})
df_2020.plot.density(figsize = (10, 7))


# ** Create a heatmap of the correlation between the stocks Close Price in each year.**
# 

# In[25]:


# Import seaborn
import seaborn as sns

#Data for 2020
Amazon_2020_close = df[(df['Company'] == 'Amazon') & (df.Date.dt.year == 2020)]['Close'].tolist()[1:]
Facebook_2020_close = df[(df['Company'] == 'Facebook') & (df.Date.dt.year == 2020)]['Close'].tolist()[1:]
Microsoft_2020_close = df[(df['Company'] == 'Microsoft') & (df.Date.dt.year == 2020)]['Close'].tolist()[1:]
Twitter_2020_close = df[(df['Company'] == 'Twitter') & (df.Date.dt.year == 2020)]['Close'].tolist()[1:]
Apple_2020_close= df[(df['Company'] == 'Apple') & (df.Date.dt.year == 2020)]['Close'].tolist()[1:]
Google_2020_close = df[(df['Company'] == 'Google') & (df.Date.dt.year == 2020)]['Close'].tolist()[1:]
df_2020 = pd.DataFrame({'Amazon': Amazon_2020_close, 'Facebook': Facebook_2020_close, 'Microsoft': Microsoft_2020_close, 'Twitter': Twitter_2020_close, 'Apple': Apple_2020_close, 'Google': Google_2020_close})
corr_1 = df_2020.corr()

#Data for 2021
Amazon_2021_close = df[(df['Company'] == 'Amazon') & (df.Date.dt.year == 2021)]['Close'].tolist()[1:]
Facebook_2021_close = df[(df['Company'] == 'Facebook') & (df.Date.dt.year == 2021)]['Close'].tolist()[1:]
Microsoft_2021_close = df[(df['Company'] == 'Microsoft') & (df.Date.dt.year == 2021)]['Close'].tolist()[1:]
Twitter_2021_close = df[(df['Company'] == 'Twitter') & (df.Date.dt.year == 2021)]['Close'].tolist()[1:]
Apple_2021_close= df[(df['Company'] == 'Apple') & (df.Date.dt.year == 2021)]['Close'].tolist()[1:]
Google_2021_close = df[(df['Company'] == 'Google') & (df.Date.dt.year == 2021)]['Close'].tolist()[1:]
df_2021 = pd.DataFrame({'Amazon': Amazon_2021_close, 'Facebook': Facebook_2021_close, 'Microsoft': Microsoft_2021_close, 'Twitter': Twitter_2021_close, 'Apple': Apple_2021_close, 'Google': Google_2021_close})
corr_2 = df_2021.corr()

#Data for 2022
Amazon_2022_close = df[(df['Company'] == 'Amazon') & (df.Date.dt.year == 2022)]['Close'].tolist()[1:]
Facebook_2022_close = df[(df['Company'] == 'Facebook') & (df.Date.dt.year == 2022)]['Close'].tolist()[1:]
Microsoft_2022_close = df[(df['Company'] == 'Microsoft') & (df.Date.dt.year == 2022)]['Close'].tolist()[1:]
Twitter_2022_close = df[(df['Company'] == 'Twitter') & (df.Date.dt.year == 2022)]['Close'].tolist()[1:]
Apple_2022_close= df[(df['Company'] == 'Apple') & (df.Date.dt.year == 2022)]['Close'].tolist()[1:]
Google_2022_close = df[(df['Company'] == 'Google') & (df.Date.dt.year == 2022)]['Close'].tolist()[1:]
df_2022 = pd.DataFrame({'Amazon': Amazon_2022_close, 'Facebook': Facebook_2022_close, 'Microsoft': Microsoft_2022_close, 'Twitter': Twitter_2022_close, 'Apple': Apple_2022_close, 'Google': Google_2022_close})
corr_3 = df_2022.corr()

fig, ax = plt.subplots(3, 1, figsize=(24,12))

for i,d in enumerate([corr_1,corr_2,corr_3]):
   
    sns.heatmap(d,
#                 cmap="viridis",  # Choose a squential colormap
                annot=True, # Label the value
                annot_kws={'fontsize':10},  # Reduce size of label to fit
                square=True,     # Force square cells
                fmt=".1f",        # One decimal place
#                 vmax=1,         # Ensure same 
#                 vmin=-1,          # color scale
                linewidth=0.01,  # Add gridlines
                linecolor="#222",# Adjust gridline color
                ax=ax[i],        # Arrange in subplot
               )
    
ax[0].set_title('2020')
ax[1].set_title('2021')
ax[2].set_title('2022')
plt.tight_layout()


# # Part 2.
# In[27]:


# In the following, we will analyze the airline stocks between Delta, Southwest and Southwest Airlines.


# In[28]:

get_ipython().system('pip install pandas-datareader')
#conda install -c anaconda pandas-datareader 
#!pip install --upgrade pandas
import pandas as pd
import numpy as np
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data, wb

import datetime
from datetime import date
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


start1 = datetime.datetime(2018, 1, 1)

end1 = date.today()


# In[30]:


# Delta
Delta = data.DataReader("DAL", 'yahoo', start1, end1)

# Southwest
Southwest = data.DataReader("LUV", 'yahoo', start1, end1)

# United
United = data.DataReader("UAL", 'yahoo', start1, end1)


# In[31]:


Delta.reset_index(inplace=True)
Southwest.reset_index(inplace=True)
United.reset_index(inplace=True)
United.head()


# In[32]:


Delta["Company"]='Delta'
Southwest["Company"]='Southwest'
United["Company"]='United'


# In[33]:


company = {'Delta','Southwest','United'}


# In[34]:


df = pd.concat([Delta, Southwest, United], ignore_index=True)


# In[35]:


import matplotlib.pyplot as plt

for i in company:
    plt.plot(df[df['Company'] == i]['Date'],df[df['Company'] == i]['Close'], label = i)
    plt.legend()

plt.xlabel('Date')
plt.ylabel('Price')

plt.tight_layout()
plt.savefig("normalvars.png", dpi=150)


# In[36]:


# The stock prices for the aviation sector decline significantly in 2020, as shown in the graph above. 
# Due to the pandemic, fewer people are traveling, which significantly reduces the demand for airplanes. 
# The stocks of airlines reflect this. 
# At the beginning of 2021, as vaccines become more widely used, demand for aircraft rises. 
# Stocks of airlines soar.


# In[37]:


for i in company:
    plt.plot(df[df['Company'] == i]['Date'],df[df['Company'] == i]['Volume'], label = i)
    plt.legend()
    
plt.xlabel('Date')
plt.ylabel('Volume')

plt.tight_layout()
plt.savefig("normalvars.png", dpi=150)


# In[38]:


# The graph displays the volume of trading for these firms and makes it very evident that United Airlines stock 
# is exchanged more frequently than those of other airlines.


# In[39]:


# Market Capitalisation
for i in company:
    plt.plot(df[df['Company'] == i]['Date'],df[df['Company'] == i]['Volume']*df[df['Company'] == i]['Open'], label = i)
    plt.legend()

plt.xlabel('Date')
plt.ylabel('Market Cap')

plt.tight_layout()
plt.savefig("normalvars.png", dpi=150)


# In[40]:


# As is evident from the graph, United Airlines appears to be traded at a premium.


# In[41]:


def MA_10(company):
    return df[df['Company'] == company]['Close'].rolling(5).mean()

def MA_60(company):
    return df[df['Company'] == company]['Close'].rolling(60).mean()

def MA_120(company):
    return df[df['Company'] == company]['Close'].rolling(120).mean()


# In[42]:


def plot_MA(company, index):
    sub = fig.add_subplot(3,2,index)
    sub.set_xlabel('date')
    sub.set_ylabel('price')
    sub.set_title(company+ ' Airlines')
    #sub.plot(find_avg_closing_price(company))
    sub.plot(df[df['Company'] == company]['Date'],MA_10(company), label = 'MA10')
    sub.plot(df[df['Company'] == company]['Date'],MA_60(company),  label = 'MA60')
    sub.plot(df[df['Company'] == company]['Date'],MA_120(company),  label = 'MA120')
    sub.legend()
    


# In[43]:


fig = plt.figure(figsize=(10,6))

index = 1
for i in company:
    plot_MA(i, index)
    index = index + 1
    
plt.tight_layout()
plt.savefig("normalvars.png", dpi=150)


# In[44]:


# According to the above table, the averages are trending downwards in the short, medium and long term. 
# This is due to the inflationary factor, 
# which reduces the willingness to travel and reduces the demand for airlines.


# In[45]:


# From the current point of view, 
# the short-term averages are breaking through the medium- and long-term averages from the bottom to the top as a golden crossover, 
# constituting a buying moment. 
# This is due to the increasing demand for airlines as people travel more towards Christmas at the end of the year.


# In[46]:


from pandas.plotting import scatter_matrix

data = pd.concat([Delta['Open'],Southwest['Open'],United['Open']],axis = 1)
data.columns = ['DALOpen','LUVOpen','UALOpen']
scatter_matrix(data, figsize = (8,8), hist_kwds= {'bins':250})


# In[47]:


# The above graph is made up of the histograms for each firm combined with a scatter plot that compares the stocks of two companies at once. 
# The graph makes it quite evident that these three equities are only loosely exhibiting a linear link with one another.


# In[48]:


df3 =  df.groupby(['Company']).Close.pct_change()
new_df = df3.to_frame()
new_df['Company'] = df['Company']
new_df['Date'] = df['Date']
new_df['Return'] = df['Close'].pct_change()
new_df.head()


# In[49]:


def plot_distribution(company,index):
    new_df[new_df['Company'] == company]['Close'].hist(bins=50, label = company)
    plt.legend()


# In[50]:


fig = plt.figure(figsize=(15,10))

index = 1
for i in company:
    plot_distribution(i, index)
    index = index + 1
    
plt.tight_layout()
plt.savefig("normalvars.png", dpi=150)


# In[51]:


# The graph makes it evident that United Airlines has the widest percentage increase in stock price histogram, 
# indicating that its stock is the most volatile when compared to the other two.


# In[52]:


# Conclusion:
# We draw the conclusion from our analysis that the pandemic has had a long-term negative impact on airline stock prices.
# However, in the near future, a fresh investment opportunity has arisen as a result of the increased demand for the aviation sector brought on by the approach of Christmas. 
# We come to the conclusion that there is a linear correlation between the stocks of Delta Air Lines, Southwest Airlines, and United Airlines. 
# The stock of United Airlines is more erratic.

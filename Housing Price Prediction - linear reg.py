#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# importing data
data = pd.read_csv("Raw_Housing_Prices.csv")
data.head()
data.info()


# In[3]:


# replacing the Numerical NaN value to string for handling the missing value
data['No of Times Visited'] = data['No of Times Visited'].fillna('NaN')


# In[4]:


data["Sale Price"].describe()


# In[5]:


# distribution of target variables
data['Sale Price'].plot.hist()


# In[6]:


# checking Quantiles
q1 = data['Sale Price'].quantile(0.25)
q3 = data['Sale Price'].quantile(0.75)
q1, q3


# In[7]:


# calculating iqr
iqr = q3 - q1
iqr


# In[8]:


upper_limit = q3 + 1.5*iqr
lower_limit = q1 - 1.5*iqr
upper_limit, lower_limit


# In[9]:


# imputing outliers
def limit_imputer(value):
    if value > upper_limit:
        return upper_limit
    if value < lower_limit:
        return lower_limit
    else:
        return value


# In[10]:


data['Sale Price'] = data['Sale Price'].apply(limit_imputer)


# In[11]:


data['Sale Price'].describe()


# In[12]:


data['Sale Price'].plot.hist()


# In[13]:


# checking missing values
data.isnull().sum()


# In[14]:


data['Sale Price'].dropna(inplace = True)
data['Sale Price'].isnull().sum()


# In[15]:


data.info()


# In[16]:


numerical_columns = ['No of Bathrooms', 'Flat Area (in Sqft)','Lot Area (in Sqft)',
                     'Area of the House from Basement (in Sqft)','Latitude',
                     'Longitude','Living Area after Renovation (in Sqft)']


# In[17]:


# imputing missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])


# In[18]:


data.info()


# In[19]:


# ZIPCODE TRANSFORM

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
data['Zipcode'] = imputer.fit_transform(data['Zipcode'].values.reshape(-1,1))


# In[20]:


data["Zipcode"].shape


# In[21]:


column = data['Zipcode'].values.reshape(-1,1)
column.shape


# In[22]:


imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
data['Zipcode'] = imputer.fit_transform(column)


# In[23]:


data.info()


# In[24]:


# OTHER TRANSFORMATION

data['No of Times Visited'].unique()


# In[25]:


#converting from string to categorical
mapping = {'NaN': '0',
          'Once': '1',
          'Twice': '2',
          'Thrice': '3',
          'Four': '4'}
data['No of Times Visited'] = data['No of Times Visited'].map(mapping)


# In[26]:


data['No of Times Visited'].unique()


# In[27]:


# new variable creation
data['Ever Renovated'] = np.where(data['Renovated Year'] == 0, 'No', 'Yes')

data.head()


# In[28]:


# manipulating datetime variable
data['Purchased Year'] = pd.DatetimeIndex(data['Date House was Sold']).year


# In[29]:


data['Years Since Renovation'] = np.where(data['Ever Renovated'] == 'Yes',
                                         abs(data['Purchased Year'] - 
                                            data['Renovated Year']), 0)


# In[30]:


data.head()


# In[31]:


# dropping redundant variable
data.drop(columns = ['Purchased Year', 'Date House was Sold', 'Renovated Year'], inplace=True)


# In[32]:


data.head()


# In[33]:


# ZIPCODE BIN
data.drop(columns = 'ID', inplace = True)


# In[34]:


data['Condition of the House'].head(10)


# In[35]:


data['Condition of the House'].value_counts()


# In[36]:


data.groupby('Condition of the House')['Sale Price'].mean().plot(kind = 'bar')


# In[37]:


data.groupby('Condition of the House')['Sale Price'].mean().sort_values().plot(kind = 'bar')


# In[38]:


data.groupby('Waterfront View')['Sale Price'].mean().sort_values().plot(kind = 'bar')


# In[39]:


data.groupby('Ever Renovated')['Sale Price'].mean().sort_values().plot(kind = 'bar')


# In[40]:


data.groupby('Zipcode')['Sale Price'].mean().sort_values().plot(kind = 'bar')


# # Linear Regression

# In[41]:


data.dropna(inplace=True)
X = data.drop(columns = ['Sale Price'])
Y = data['Sale Price']


# In[42]:


# variable Transformation
# checking distribution of independent numerical variable
def distribution(data, var):
    plt.figure(figsize = (len(var)*6,6), dpi = 120)
    for j,i in enumerate(var):
        plt.subplot(1,len(var),j+1)
        plt.hist(data[i])
        plt.title(i)


# In[43]:


numerical_columns = ['No of Bedrooms', 'No of Bathrooms', 'Lot Area (in Sqft)',
       'No of Floors',
       'Area of the House from Basement (in Sqft)', 'Basement Area (in Sqft)',
       'Age of House (in Years)', 'Latitude', 'Longitude',
       'Living Area after Renovation (in Sqft)',
       'Lot Area after Renovation (in Sqft)',
       'Years Since Renovation']


# In[44]:


for i in numerical_columns:
    X[i] = pd.to_numeric(X[i])


# In[45]:


distribution(X, numerical_columns)


# In[46]:


# removing right Skew
def right_skew(x):
    return np.log(abs(x+500))
right_skew_variables = ['No of Bedrooms', 'No of Bathrooms', 'Lot Area (in Sqft)',
       'No of Floors',
       'Area of the House from Basement (in Sqft)', 'Basement Area (in Sqft)',
        'Longitude',
       'Living Area after Renovation (in Sqft)',
       'Lot Area after Renovation (in Sqft)',
       'Years Since Renovation']


# In[47]:


for i in right_skew_variables:
    X[i] = X[i].map(right_skew)
    
# removing infinite values
X = X.replace(np.inf, np.nan)
X.dropna(inplace=True)


# In[48]:


distribution(X, numerical_columns)


# In[49]:


# SCALING THE DATATSET
X.head()


# In[50]:


X["Waterfront View"] = X["Waterfront View"].map({    'No':0,
   'Yes':1
})


X['Condition of the House'] = X['Condition of the House'].map({'Bad':1,
                                                                     'Okay':2,
                                                                     'Fair':3,
                                                                     'Good':4,
                                                                     'Excellent':5
})

X['Ever Renovated'] = X['Ever Renovated'].map({
    'No':0,
    'Yes':1
})

X.head()


# In[51]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Y = data['Sale Price']
X1 = scaler.fit_transform(X)
X = pd.DataFrame(data = X1, columns = X.columns)
X.head()


# In[52]:


# checking and removing multi-collinearity
X.corr()


# In[53]:


# pair of independent variables with correlation greater than 0.5
k = X.corr()
z = [[str(i),str(j)] for i in k.columns for j in k.columns if (k.loc[i,j]>abs(0.5))&(i!=j)]
z, len(z)


# In[54]:


# CALCULATING VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
vif_data = X[:]

# calculating VIF for every column
VIF = pd.Series([vif(vif_data.values,i) for i in range(vif_data.shape[1])], index = vif_data.columns)
VIF


# In[55]:


def mc_remover(data):
    vif_ = pd.Series([vif(data.values, i) for i in range(data.shape[1])], index = data.columns)
    if vif_.max() > 5:
        print(vif_[vif_ == vif_.max()].index[0],'has been removed')
        data = data.drop(columns = [vif_[vif_ == vif_.max()].index[0]])
        return data
    else:
        print('No Multicollinearity present anymore')
        return data


# In[56]:


for i in range(7):
    vif_data = mc_remover(vif_data)
vif_data.head()


# In[57]:


# Remaining Columns
VIF = pd.Series([vif(vif_data.values, i) for i in range(vif_data.shape[1])], index = vif_data.columns)
VIF, len(vif_data.columns)


# In[58]:


X = vif_data[:]


# In[59]:


#TEST/TEST SET
Y = data["Sale Price"]

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(X, Y, test_size = 0.3, random_state = 101)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[60]:


# training
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)


# In[61]:


print(lr.coef_)
print()
print(lr.intercept_)


# In[62]:


predictions = lr.predict(x_test)


# In[63]:


lr.score(x_test, y_test)


# In[64]:


#RESIDUALS
residuals = predictions - y_test

residual_table = pd.DataFrame({'residuals':residuals,
                    'predictions':predictions})
residual_table = residual_table.sort_values( by = 'predictions')


z = [i for i in range(int(residual_table['predictions'].max()))]
k = [0 for i in range(int(residual_table['predictions'].max()))]


# In[65]:


plt.figure(dpi = 130, figsize = (17,7))

plt.scatter( residual_table['predictions'], residual_table['residuals'], color = 'red', s = 2)
plt.plot(z, k, color = 'green', linewidth = 3, label = 'regression line')
plt.ylim(-800000, 800000)
plt.xlabel('fitted points (ordered by predictions)')
plt.ylabel('residuals')
plt.title('residual plot')
plt.legend()
plt.show()


# In[66]:


#DISTRIBUTION OF ERRORS
plt.figure(dpi = 100, figsize = (10,7))
plt.hist(residual_table['residuals'], color = 'red', bins = 200)
plt.xlabel('residuals')
plt.ylabel('frequency')
plt.title('distribution of residuals')
plt.show()


# # MODEL COEFFICIENTS

# In[67]:


coefficients_table = pd.DataFrame({'column': x_train.columns,
                                  'coefficients': lr.coef_})
coefficient_table = coefficients_table.sort_values(by = 'coefficients')


plt.figure(figsize=(8, 6), dpi=120)
x = coefficient_table['column']
y = coefficient_table['coefficients']
plt.barh( x, y)
plt.xlabel( "Coefficients")
plt.ylabel('Variables')
plt.title('Normalized Coefficient plot')
plt.show()


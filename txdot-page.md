---
layout: default
---
[home](./)

## Welcome to my TxDot project

##### This was originally done as my capstone project for a Data Science course I took with General Assembly in 2016

### Background:

 Each month, The Texas Department of Transportation auctions off construction projects that bid on by contracting companies. These projects are paid for by the Texas Government and so the data is freely available through an open data request. Each project can cost multiple millions of dollars so winning a bid means a lot of business to the company and a large cost to the government. Therefore, estimating the winning bid price is important for both sides.

 TxDOT wants to make sure to budget enough money for each project and relies on engineers to estimate the cost according to the equipment and supplies needed in each project. These estimates are usually based off the average going price of each item and then summed together. TxDOT judges these estimates base on how close they are to the actual winning bid. Their goal is to be within 10% of the actual winning bid and are investigated if otherwise.

 The goal of this project was to estimate new project winning prices using the historical price estimates, project descriptions, and winning bid prices.

### Summary:

 In my testing set, I was able to increase the accuracy of the TxDOT estimates by about 7% (according to the "within 10%" goal that they use to judge themselves on). However, overall my estimates were more inaccurate in terms of total cost.

--------
--------

### Import Modules

```python
import os
import math
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt

import csv
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import feature_selection, linear_model
```

### Load in the Data


```python
df = pd.read_csv('C:/Users/Collin/Documents/Python_Projects/TxDOT/data/Bid_Data.csv')
len(df)
```




    32447



### Formatting & Identifying the Data


```python
df = df[df['Rank'] == 1]
df = df[df['Type'] == 'Construction']

df = df.drop(' From', 1)
df = df.drop('To', 1)
df = df.drop('Contract Description', 1)
df = df.drop('Contractor', 1)
df = df.drop('Contract Category', 1)
df = df.drop('LNG MON', 1)
df = df.drop('MONTH', 1)

df['Award Amount'] = df['Award Amount'].str.lstrip('$')
df['Engineers Estimate'] = df['Engineers Estimate'].str.lstrip('$')
df['Award Amount'] = df['Award Amount'].str.replace(',','').astype(float)
df['Engineers Estimate'] = df['Engineers Estimate'].str.replace(',','').astype(float)

#Renaming Variables
df['EngEst'] = df['Engineers Estimate']
df['NBidders'] = df['Number of Bidders']
df['Date'] = pd.to_datetime(df['Letting Date'])
df.set_index('Date' , inplace=True)
df['Year'] = df.index.year
df['Month'] = df.index.month
df['WinBid'] = df['Award Amount']

# Creating New Varialbes
df['Diff'] = df['EngEst'] - df['WinBid']
df['lnWinBid'] = np.log(df['WinBid'])
df['lnEngEst'] = np.log(df['EngEst'])
df['DiffLn'] = df['lnWinBid'] - df['lnEngEst']
df['Within10Percent'] = 1
df['PercentOff'] = df['Diff'] / df['EngEst']
df['MoreOrLessThan10'] = 0
df['LessThan10'] = 0
df['MoreThan10'] = 0

df.loc[(df.PercentOff > .10) , 'Within10Percent'] = 0
df.loc[(df.PercentOff < -.10) , 'Within10Percent'] = 0
df.loc[(df.PercentOff > .10) , 'MoreOrLessThan10'] = 1
df.loc[(df.PercentOff < -.10) , 'MoreOrLessThan10'] = 2
df.loc[(df.PercentOff > .10) , 'MoreThan10'] = 1
df.loc[(df.PercentOff < -.10) , 'LessThan10'] = 1

print len(df)
```

    5177


### Exploring the Data, We see that Bids are Log Normally Distributed


```python
sns.jointplot(x="EngEst", y="WinBid", data=df, kind="reg"); sns.jointplot(x="lnEngEst", y="lnWinBid", data=df, kind="reg");
```


![png](https://mathtodon.github.io/assets/images/output_7_0.png)



![png](https://mathtodon.github.io/assets/images/output_7_1.png)


Because the TxDot Estimates are so highly correlated with the winning bid, we can use it as a baseline for the model predictions

My idea was to classify the TxDot Estimates according to their scoring method of estimating within 10% of the winning bid and then having 3 separate regressive models for the 3 classes (within 10%, more than 10%, less than 10%).

TxDot's goal for their estimates is to have 55% of their estimates within 10% of the winning bid.

TxDot considers bids over or under the estimate by more than 10% as bad

We can then use this score to judge our model predictions


```python
#Using ALL the Data

Percent = float(df.Within10Percent.sum()) / len(df)
print  (Percent)*100 , '% of All the TxDOT estimates were within 10% of actual bid'

Percent_April_2016 = float(df[(df.Year == 2016) & (df.Month == 4)].Within10Percent.sum()) / len(df_test)
print  (Percent_April_2016)*100 , '% of the April 2016 TxDOT estimates were within 10% of actual bid'
```


    50.3 % of All the TxDOT estimates were within 10% of actual bid
    46.3 % of the April 2016 TxDOT estimates were within 10% of actual bid


### We now need to build and train a model that can classify a bid as over or under the TxDOT estimate

#### Here We can see how this classification might look


```python
cmap = {'0': 'g', '1': 'r', '2': 'b' }
df['cMoreOrLessThan10'] = df.MoreOrLessThan10.apply(lambda x: cmap[str(x)])
print df.plot('EngEst', 'WinBid', kind='scatter', c=df.cMoreOrLessThan10)
```




![png](https://mathtodon.github.io/assets/images/output_12_1.png)



```python
cmap = {'0': 'g', '1': 'r', '2': 'b' }
df['cMoreOrLessThan10'] = df.MoreOrLessThan10.apply(lambda x: cmap[str(x)])
print df.plot('lnEngEst', 'lnWinBid', kind='scatter', c=df.cMoreOrLessThan10)
```



![png](https://mathtodon.github.io/assets/images/output_13_1.png)


### Splitting the Data into Training and Testing Sets


```python
df_test = df[(df.Year == 2016) & (df.Month == 4)]
print len(df_test) , 'projects in April 2016'

df_train = df[(df.Year != 2016) | (df.Month != 4)]
print len(df_train) ,'projects from Jan 2010 to April 2016'

#df_train[['Year','Month']].tail()
```


    67 projects in April 2016
    5110 projects from Jan 2010 to April 2016



```python
#df_test.columns
```

#### This will split the training and testing sets into a more useful format for loading into models


```python
names_X = ['Length','NBidders','Year','Month','lnEngEst','Time']

def X_y(df):
    X = df[ names_X ]
    y_more = df['MoreThan10']
    y_less =df['LessThan10']
    return X, y_more, y_less

train_X, train_y_more, train_y_less = X_y(df_train)
test_X, test_y_more, test_y_less = X_y(df_test)

print len(train_y_more)
print len(train_y_less)
print len(test_y_more)
print len(test_y_less)
```

    5110
    5110
    67
    67



```python
test_X.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Length</th>
      <th>NBidders</th>
      <th>Year</th>
      <th>Month</th>
      <th>lnEngEst</th>
      <th>Time</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-04-05</th>
      <td>15.055</td>
      <td>2</td>
      <td>2016</td>
      <td>4</td>
      <td>14.979827</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>0.016</td>
      <td>5</td>
      <td>2016</td>
      <td>4</td>
      <td>14.722602</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>2.024</td>
      <td>4</td>
      <td>2016</td>
      <td>4</td>
      <td>16.836501</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>6.294</td>
      <td>3</td>
      <td>2016</td>
      <td>4</td>
      <td>14.893941</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>0.455</td>
      <td>3</td>
      <td>2016</td>
      <td>4</td>
      <td>13.837735</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


### Probabilistic Classifying using Logistic Regression

### More Than 10%

#### Training


```python
model_1 = linear_model.LogisticRegression()
model_1.fit(train_X, train_y_more)

print 'intercept    =', model_1.intercept_
print 'coefficients =', model_1.coef_
```

    intercept    = [ 0.0001453]
    coefficients = [[ 0.00230722  0.21597156 -0.00191552  0.0116006   0.15382005 -0.00279872]]


#### Scoring


```python
print 'correct training classification = ', model_1.score(train_X, train_y_more)
print 'correct testing classification = ', model_1.score(test_X, test_y_more)
```

    correct training classification =  0.714677103718
    correct testing classification =  0.65671641791


#### Predicting


```python
y_logit_more = model_1.intercept_ + (test_X * model_1.coef_[0]).sum(axis = 1)
y_odds_more = np.exp(y_logit_more)
y_p_more = y_odds_more / (1 + y_odds_more)
y_p_more.head()
```




    Date
    2016-04-05    0.260327
    2016-04-06    0.384470
    2016-04-06    0.411735
    2016-04-06    0.296980
    2016-04-06    0.261602
    dtype: float64




```python
df_test.loc[:,'p_more'] = y_p_more
```

### Less Than 10%

#### Training


```python
model_2 = linear_model.LogisticRegression()
model_2.fit(train_X, train_y_less)

print 'intercept    =', model_2.intercept_
print 'coefficients =', model_2.coef_
```

    intercept    = [-0.00046124]
    coefficients = [[-0.01289253 -0.40220173  0.00238074 -0.01342247 -0.32558187  0.00243721]]


#### Scoring


```python
print 'correct training classification = ', model_2.score(train_X, train_y_less)
print 'correct testing  classification = ', model_2.score(test_X, test_y_less)
```

    correct training classification =  0.791585127202
    correct testing  classification =  0.731343283582


#### Predicting


```python
y_logit_less = model_2.intercept_ + (test_X * model_2.coef_[0]).sum(axis = 1)
y_odds_less = np.exp(y_logit_less)
y_p_less = y_odds_less / (1 + y_odds_less)
y_p_less.head()
```




    Date
    2016-04-05    0.244116
    2016-04-06    0.113125
    2016-04-06    0.085400
    2016-04-06    0.199164
    2016-04-06    0.274409
    dtype: float64




```python
df_test.loc[:,'p_less'] = y_p_less
```

### Within 10%

#### P(within) = 1- P(more) - P(less)


```python
y_p_within = 1-y_p_more - y_p_less

y_p_within.head()
```




    Date
    2016-04-05    0.495556
    2016-04-06    0.502405
    2016-04-06    0.502865
    2016-04-06    0.503856
    2016-04-06    0.463989
    dtype: float64




```python
df_test.loc[:,'p_within'] = y_p_within
```

### Build a model given that the Winning Bid is within 10% of TxDot Estimate


```python
#subsets the training data to just those who were within 10% of the TxDot Estimate
df_train_within = df[df.Within10Percent == 1]

model_3 = smf.ols(formula = 'lnWinBid ~ lnEngEst+Year+Month+Year*Month+NBidders+NBidders*Year+Time', data = df_train_within).fit()
model_3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>lnWinBid</td>     <th>  R-squared:         </th> <td>   0.999</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.999</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>2.592e+05</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 01 Jun 2016</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>17:52:08</td>     <th>  Log-Likelihood:    </th> <td>  3907.2</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2606</td>      <th>  AIC:               </th> <td>  -7798.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2598</td>      <th>  BIC:               </th> <td>  -7752.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th>
</tr>
<tr>
  <th>Intercept</th>     <td>   -3.2112</td> <td>    3.597</td> <td>   -0.893</td> <td> 0.372</td> <td>  -10.265     3.843</td>
</tr>
<tr>
  <th>lnEngEst</th>      <td>    0.9956</td> <td>    0.001</td> <td>  956.110</td> <td> 0.000</td> <td>    0.994     0.998</td>
</tr>
<tr>
  <th>Year</th>          <td>    0.0016</td> <td>    0.002</td> <td>    0.909</td> <td> 0.363</td> <td>   -0.002     0.005</td>
</tr>
<tr>
  <th>Month</th>         <td>    0.2336</td> <td>    0.364</td> <td>    0.641</td> <td> 0.521</td> <td>   -0.481     0.948</td>
</tr>
<tr>
  <th>Year:Month</th>    <td>   -0.0001</td> <td>    0.000</td> <td>   -0.641</td> <td> 0.522</td> <td>   -0.000     0.000</td>
</tr>
<tr>
  <th>NBidders</th>      <td>   -0.2604</td> <td>    0.545</td> <td>   -0.478</td> <td> 0.633</td> <td>   -1.328     0.808</td>
</tr>
<tr>
  <th>NBidders:Year</th> <td>    0.0001</td> <td>    0.000</td> <td>    0.471</td> <td> 0.638</td> <td>   -0.000     0.001</td>
</tr>
<tr>
  <th>Time</th>          <td> 4.125e-05</td> <td> 8.94e-06</td> <td>    4.613</td> <td> 0.000</td> <td> 2.37e-05  5.88e-05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>576.150</td> <th>  Durbin-Watson:     </th> <td>   1.925</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 121.613</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.183</td>  <th>  Prob(JB):          </th> <td>3.91e-27</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.007</td>  <th>  Cond. No.          </th> <td>5.79e+07</td>
</tr>
</table>



#### Predicting


```python
df_test.loc[:,'Lnprediction_within'] = model_3.predict(test_X)
df_test.Lnprediction_within.head()
```




    Date
    2016-04-05    14.972721
    2016-04-06    14.706426
    2016-04-06    16.814502
    2016-04-06    14.883813
    2016-04-06    13.832213
    Name: Lnprediction_within, dtype: float64



### Build a model given that the Winning Bid is More Than 10% of TxDot Estimate


```python
#subsets the training data to just those who were more than 10% of the TxDot Estimate
df_train_more = df[df.MoreThan10 == 1]

model_4 = smf.ols(formula = 'lnWinBid ~ lnEngEst+Year+Month+Year*Month+NBidders+NBidders*Year+Time', data = df_train_more).fit()
model_4.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>lnWinBid</td>     <th>  R-squared:         </th> <td>   0.992</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.992</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>2.733e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 01 Jun 2016</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>17:52:22</td>     <th>  Log-Likelihood:    </th> <td>  1033.0</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1488</td>      <th>  AIC:               </th> <td>  -2050.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1480</td>      <th>  BIC:               </th> <td>  -2008.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th>
</tr>
<tr>
  <th>Intercept</th>     <td>  -13.2256</td> <td>   10.209</td> <td>   -1.296</td> <td> 0.195</td> <td>  -33.250     6.799</td>
</tr>
<tr>
  <th>lnEngEst</th>      <td>    1.0004</td> <td>    0.003</td> <td>  332.010</td> <td> 0.000</td> <td>    0.995     1.006</td>
</tr>
<tr>
  <th>Year</th>          <td>    0.0065</td> <td>    0.005</td> <td>    1.273</td> <td> 0.203</td> <td>   -0.003     0.016</td>
</tr>
<tr>
  <th>Month</th>         <td>   -0.5925</td> <td>    0.938</td> <td>   -0.632</td> <td> 0.528</td> <td>   -2.432     1.247</td>
</tr>
<tr>
  <th>Year:Month</th>    <td>    0.0003</td> <td>    0.000</td> <td>    0.633</td> <td> 0.527</td> <td>   -0.001     0.001</td>
</tr>
<tr>
  <th>NBidders</th>      <td>    3.0432</td> <td>    1.479</td> <td>    2.057</td> <td> 0.040</td> <td>    0.141     5.945</td>
</tr>
<tr>
  <th>NBidders:Year</th> <td>   -0.0015</td> <td>    0.001</td> <td>   -2.060</td> <td> 0.040</td> <td>   -0.003 -7.25e-05</td>
</tr>
<tr>
  <th>Time</th>          <td>  8.62e-05</td> <td> 2.87e-05</td> <td>    3.000</td> <td> 0.003</td> <td> 2.98e-05     0.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>744.819</td> <th>  Durbin-Watson:     </th> <td>   1.877</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>5533.375</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-2.224</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>11.335</td>  <th>  Cond. No.          </th> <td>5.77e+07</td>
</tr>
</table>



#### Predicting


```python
df_test.loc[:,'Lnprediction_more'] = model_4.predict(test_X)
```


### Build a model given that the Winning Bid is Less Than 10% of TxDot Estimate


```python
#subsets the training data to just those who were less than 10% of the TxDot Estimate
df_train_less = df[df.LessThan10 == 1]

model_5 = smf.ols(formula = 'lnWinBid ~ lnEngEst+Year+Month+Year*Month+NBidders+NBidders*Year+Time', data = df_train_less).fit()
model_5.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>lnWinBid</td>     <th>  R-squared:         </th> <td>   0.995</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.995</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>2.895e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 01 Jun 2016</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>17:52:27</td>     <th>  Log-Likelihood:    </th> <td>  1000.8</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1083</td>      <th>  AIC:               </th> <td>  -1986.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1075</td>      <th>  BIC:               </th> <td>  -1946.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th>
</tr>
<tr>
  <th>Intercept</th>     <td>  -33.7856</td> <td>   10.879</td> <td>   -3.106</td> <td> 0.002</td> <td>  -55.131   -12.440</td>
</tr>
<tr>
  <th>lnEngEst</th>      <td>    0.9719</td> <td>    0.003</td> <td>  310.603</td> <td> 0.000</td> <td>    0.966     0.978</td>
</tr>
<tr>
  <th>Year</th>          <td>    0.0171</td> <td>    0.005</td> <td>    3.161</td> <td> 0.002</td> <td>    0.006     0.028</td>
</tr>
<tr>
  <th>Month</th>         <td>    0.3353</td> <td>    1.180</td> <td>    0.284</td> <td> 0.776</td> <td>   -1.981     2.651</td>
</tr>
<tr>
  <th>Year:Month</th>    <td>   -0.0002</td> <td>    0.001</td> <td>   -0.283</td> <td> 0.777</td> <td>   -0.001     0.001</td>
</tr>
<tr>
  <th>NBidders</th>      <td>    6.5230</td> <td>    1.825</td> <td>    3.574</td> <td> 0.000</td> <td>    2.942    10.104</td>
</tr>
<tr>
  <th>NBidders:Year</th> <td>   -0.0032</td> <td>    0.001</td> <td>   -3.581</td> <td> 0.000</td> <td>   -0.005    -0.001</td>
</tr>
<tr>
  <th>Time</th>          <td>    0.0001</td> <td>    3e-05</td> <td>    3.943</td> <td> 0.000</td> <td> 5.95e-05     0.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>473.873</td> <th>  Durbin-Watson:     </th> <td>   1.994</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>3314.246</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.872</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>10.709</td>  <th>  Cond. No.          </th> <td>5.76e+07</td>
</tr>
</table>



#### Predicting


```python
df_test.loc[:,'Lnprediction_less'] = model_5.predict(test_X)
```

### Our Model Prediction of Winning Bid will be a Weighted average of Bid predictions with weights being the probability of being classified as such class found from Logistic Regression


```python
df_test['Hyp_More'] = 1

df_test['Hyp_Less'] = 1

df_test['Hyp_Within'] = 1
```


```python
df_test[['p_more', 'p_less', 'p_within', 'Lnprediction_within', 'Lnprediction_more', 'Lnprediction_less']].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p_more</th>
      <th>p_less</th>
      <th>p_within</th>
      <th>Lnprediction_within</th>
      <th>Lnprediction_more</th>
      <th>Lnprediction_less</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-04-05</th>
      <td>0.260327</td>
      <td>0.244116</td>
      <td>0.495556</td>
      <td>14.972721</td>
      <td>14.779110</td>
      <td>15.183437</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>0.384470</td>
      <td>0.113125</td>
      <td>0.502405</td>
      <td>14.706426</td>
      <td>14.487806</td>
      <td>14.866475</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>0.411735</td>
      <td>0.085400</td>
      <td>0.502865</td>
      <td>16.814502</td>
      <td>16.613899</td>
      <td>16.943218</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>0.296980</td>
      <td>0.199164</td>
      <td>0.503856</td>
      <td>14.883813</td>
      <td>14.681865</td>
      <td>15.077643</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>0.261602</td>
      <td>0.274409</td>
      <td>0.463989</td>
      <td>13.832213</td>
      <td>13.625224</td>
      <td>14.051156</td>
    </tr>
  </tbody>
</table>
</div>



### E[Bid] = P(within10)xE[within10] + P(above10)xE[above10] + P(below10)xE[below10]


```python
df_test.loc[:,'lnpred'] = df_test.p_within*df_test.Lnprediction_within + df_test.p_more*df_test.Lnprediction_more + df_test.p_less*df_test.Lnprediction_less

df_test.loc[:,'BidPrediction'] = np.exp(df_test.loc[:,'lnpred'])  
df_test.loc[:,'PredDiff'] = df_test.loc[:,'BidPrediction'] - df_test.loc[:,'WinBid']
df_test.loc[:,'PredPercentOff'] = df_test.loc[:,'PredDiff'] / df_test.loc[:,'BidPrediction']

df_test.loc[:,'PredWithin10Percent'] = 1
df_test.loc[(df_test.PredPercentOff > .10) , 'PredWithin10Percent'] = 0
df_test.loc[(df_test.PredPercentOff < -.10) , 'PredWithin10Percent'] = 0
```




```python
ModelPercent = float(df_test.PredWithin10Percent.sum()) / len(df_test)
PercentIncrease = (ModelPercent)*100 - (Percent_April_2016)*100
NumberCorrectIncrease = (PercentIncrease/100)*len(df_test)
print  (Percent_April_2016)*100 , '% of the TxDOT estimates were within 10% of actual bid'
print  (ModelPercent)*100 , '% of the Model predictions were within 10% of actual bid'
print
print 'this is a increase of :', PercentIncrease, '%'
print 'or', NumberCorrectIncrease, 'more estimates within the 10% threshhold'
```

    46.2686567164 % of the TxDOT estimates were within 10% of actual bid
    53.7313432836 % of the Model predictions were within 10% of actual bid

    this is a increase of : 7.46268656716 %
    or 5.0 more estimates within the 10% threshhold



```python
print 'In April 2016 TxDOT under estimated bids by: ' , df_test.Diff.sum()
print
print 'In April 2016 the Model under estimated bids by: ' ,df_test.PredDiff.sum()
print
print 'In April 2016 the model was ' , df_test.Diff.sum() - df_test.PredDiff.sum() , 'closer to the winning bids than TxDOT'
print
print 'The model predicted a sum of' ,df_test.BidPrediction.sum() ,'for all the projects in April 2016'
print
print 'TxDOT predicted a sum of' ,df_test.EngEst.sum() ,'for all the projects in April 2016'
```

    In April 2016 TxDOT under estimated bids by:  -3410280.24

    In April 2016 the Model under estimated bids by:  -19260901.731

    In April 2016 the model was  15850621.491 closer to the winning bids than TxDOT

    The model predicted a sum of 209008961.949 for all the projects in April 2016

    TxDOT predicted a sum of 224859583.44 for all the projects in April 2016



```python
df_test[['Diff','PredDiff']].std()
```




    Diff        659308.564868
    PredDiff    857344.883547
    dtype: float64




```python
df_test[['Diff','PredDiff']].describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Diff</th>
      <th>PredDiff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.700000e+01</td>
      <td>6.700000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-5.089971e+04</td>
      <td>-2.874761e+05</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.593086e+05</td>
      <td>8.573449e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.178934e+06</td>
      <td>-5.011753e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.510351e+05</td>
      <td>-3.104509e+05</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.591124e+04</td>
      <td>-7.131061e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.269111e+05</td>
      <td>4.398933e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.390280e+06</td>
      <td>9.164722e+05</td>
    </tr>
  </tbody>
</table>
</div>



# Bid Model Comparison by Graph

#### Red is Bad  (beyond the 10% threshhold)

#### Green is Good   (within 10% of actual)

### TxDOT Estimates


```python
cmap = {'0': 'r', '1': 'g' }
df_test.loc[:,'cWithin10Percent'] = df_test.Within10Percent.apply(lambda x: cmap[str(x)])
print df_test.plot('lnEngEst', 'lnWinBid', kind='scatter', c=df_test.cWithin10Percent)
```


![png](https://mathtodon.github.io/assets/images/output_68_1.png)


### Model Predictions


```python
predcmap = {'0': 'r', '1': 'g' }
df_test.loc[:,'cPredWithin10Percent'] = df_test.PredWithin10Percent.apply(lambda x: predcmap[str(x)])
print df_test.plot('lnpred', 'lnWinBid', kind='scatter', c=df_test.cPredWithin10Percent)
```




![png](https://mathtodon.github.io/assets/images/output_70_1.png)




[home](./)
